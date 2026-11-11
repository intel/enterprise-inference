"""Unit tests for behaviour the parity suite cannot observe, mostly fixes over the legacy CLI."""
from __future__ import annotations

import json
import os
import re
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import cli, config, download, k8s, nri_policy, policy, routing  # noqa: E402
from core.config import Catalog, Model  # noqa: E402
from core.policy import _whole_cpus  # noqa: E402
from core.renderer import dedupe_args, engine_args  # noqa: E402


@pytest.fixture
def kubectl(monkeypatch):
    """Record kubectl calls as (argv, stdin); every call succeeds with empty output."""
    calls = []

    def run(*args, stdin=None):
        calls.append((args, stdin))
        return subprocess.CompletedProcess(args, 0, "", "")
    monkeypatch.setattr(k8s, "run", run)
    return calls


def test_arg_values_taken_verbatim():
    # Legacy dropped the first --arg (jq without -n); argparse would reject a value starting with --.
    o = cli.parse_args(["deploy", "m", "--arg", "--max-model-len=4096", "--arg", "--enforce-eager"])
    assert o.arg == ["--max-model-len=4096", "--enforce-eager"] and o.positional == ["deploy", "m"]


@pytest.mark.parametrize("args, tp", [(["--tensor-parallel-size", "2"], 2), (["--tensor-parallel-size=4"], 4), ([], 1)])
def test_tp_from_engine_args(args, tp):
    assert cli._engine_cli_args(cli.Options(arg=args)) == (args, tp)


def test_tp_flag_is_injected_once():
    assert cli._engine_cli_args(cli.Options(tp="2")) == (["--tensor-parallel-size", "2"], 2)


def test_dedupe_keeps_last_occurrence():
    assert dedupe_args(["--a", "1", "--b", "--a=2", "x"]) == ["--b", "--a=2", "x"]


def test_numeric_catalog_args_are_rendered():
    # Legacy jq failed on a YAML number and rendered `args: []`.
    assert engine_args(Model(name="m", namespace="n", args=["--max-model-len", 2048])) == ["--max-model-len", "2048"]


@pytest.mark.parametrize("cpu, whole", [("8", 8), ("7.5", 8), ("7500m", 8), ("0m", 0), ("abc", 0), ("²", 0)])
def test_whole_cpus(cpu, whole):
    assert _whole_cpus(cpu) == whole


@pytest.mark.parametrize("hide_ht, tp, want", [("false", 1, 28), ("false", 2, 56), ("true", 1, 14)])
def test_auto_cpu_sizes_to_smallest_node(monkeypatch, hide_ht, tp, want):
    monkeypatch.setenv("MM_NRI_HIDE_HT", hide_ht)  # hidden: the request doubles, so half the free CPUs
    monkeypatch.setattr(nri_policy, "_schedulable_nri_nodes", lambda _: ["w1", "w2"])
    monkeypatch.setattr(nri_policy, "_balloon_pods_cpu_on_node", lambda n: {"w1": 8, "w2": 0}[n])
    monkeypatch.setattr(nri_policy, "_per_numa_capacity", lambda _: [64, 64])
    monkeypatch.setattr(nri_policy, "_reserved_cpus_count", lambda _: 64)
    assert nri_policy.auto_cpu(tp) == want  # w1 (28 free per NUMA node) decides


def test_cpu_auto_keeps_live_request_without_ht_doubling(monkeypatch):
    monkeypatch.setenv("MM_NRI_HIDE_HT", "true")
    monkeypatch.setattr(k8s, "out", lambda *args: "16")
    m = Model(name="m", namespace="n", kind="LLMInferenceService", cpu="auto")
    policy.resolve_auto_cpu(policy.NriBalloons(), m, 1)
    assert m.cpu == "8"  # renders the live request of 16 again instead of doubling it on every re-deploy


@pytest.mark.parametrize("name, model_id", [("ok", "org/m'; rm -rf /models #"), ("Bad_Name", "org/m")])
def test_unsafe_names_rejected(name, model_id):
    with pytest.raises(SystemExit):
        Model(name=name, namespace="n", model_id=model_id).check()


def test_unsafe_draft_id_rejected():
    with pytest.raises(SystemExit):
        Model(name="m", namespace="n", model_id="org/m", draft_model_id="org/d'; rm -rf /models #").check()


def test_gated_draft_needs_token(kubectl, monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(download, "_curl_ok", lambda *_: True)
    monkeypatch.setattr(download, "hf_model_info", lambda repo, _: {"gated": "manual"} if repo == "org/draft" else {})
    m = Model(name="m", namespace="n", model_id="org/m", draft_model_id="org/draft", runtime="vllm",
              runtime_image="img")
    with pytest.raises(SystemExit):
        download.download_weights(m, Catalog(cli.MM_DIR / "models.yaml"), [])
    assert "Model 'org/draft' is gated" in capsys.readouterr().err and not kubectl


def test_hf_token_only_on_stdin(kubectl, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setattr(download, "_curl_ok", lambda *_: True)
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    m = Model(name="m", namespace="n", model_id="org/m", runtime="vllm", runtime_image="img", cpu="8", memory="8Gi")
    download.download_weights(m, Catalog(cli.MM_DIR / "models.yaml"), [])
    assert not any("hf_secret" in " ".join(args) for args, _ in kubectl)
    assert any(stdin == "hf_secret" for _, stdin in kubectl)


def test_litellm_key_only_on_stdin(kubectl, monkeypatch):
    monkeypatch.setattr(routing, "_litellm_key", lambda: "sk-master")
    routing._litellm("GET", "http://litellm/model/info")
    (args, stdin), = kubectl
    assert "sk-master" not in " ".join(args) and stdin == "Authorization: Bearer sk-master\n"


def test_cpu_collisions_check_runs_report(monkeypatch):
    seen = []
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, **_: seen.append(cmd) or subprocess.CompletedProcess(cmd, 3))
    monkeypatch.setenv("MM_CONFIG", str(cli.MM_DIR / "models.yaml"))
    with pytest.raises(SystemExit) as exit_:
        cli.cpu_collisions_check(cli.Options(), "")
    assert exit_.value.code == 3 and seen[0][1:] == [str(cli.MM_DIR / "scripts" / "nri-cpu-report.py"),
                                                       "--namespace", "llm-inference"]


def test_chat_template_file_absolute_path(tmp_path):
    template = tmp_path / "t.jinja"
    template.write_text("{{ messages }}\n\n")
    catalog = tmp_path / "models.yaml"
    catalog.write_text(f"models:\n  - name: m\n    model_id: org/m\n    chat_template_file: {template}\n")
    m = config._from_catalog(Catalog(catalog), "m", cli.Options())
    assert m.chat_template == "{{ messages }}"  # used like inline Jinja, trailing newlines dropped


@pytest.mark.parametrize("flag", ["--tp", "--replicas", "--wait-timeout", "--min-replicas", "--max-replicas",
                                  "--scale-target"])
def test_zero_counts_rejected(flag):
    with pytest.raises(SystemExit):
        cli.parse_args(["deploy", "m", flag, "0"])


def test_hf_token_secret_failure_aborts(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setattr(download, "_curl_ok", lambda *_: True)
    monkeypatch.setattr(k8s, "run", lambda *args, stdin=None: subprocess.CompletedProcess(args, 1, "", "forbidden"))
    m = Model(name="m", namespace="n", model_id="org/m", runtime="vllm", runtime_image="img")
    with pytest.raises(SystemExit):  # rather than a Job that waits 7200s for a missing Secret
        download.download_weights(m, Catalog(cli.MM_DIR / "models.yaml"), [])


def test_gateway_defaults(monkeypatch):
    for var in ("MM_INFERENCE_GATEWAY", "MM_INFERENCE_GATEWAY_NS", "MM_INFERENCE_GATEWAY_SVC_NS"):
        monkeypatch.delenv(var, raising=False)
    assert (config.gateway_ref(), config.gateway_service()) == (
        ("ai-gateway", "llm-inference"), ("ai-gateway", "envoy-gateway-system"))
    monkeypatch.setenv("MM_INFERENCE_GATEWAY_NS", "gw")  # legacy: one variable set both
    assert config.gateway_ref()[1] == config.gateway_service()[1] == "gw"


def test_one_reconciler_per_model(monkeypatch, tmp_path):
    held = []  # the child keeps the locked log open; a dup of the fd stands in for it here
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, stdout, **k: held.append(os.dup(stdout.fileno())))
    monkeypatch.setenv("MM_RECONCILE_LOG_DIR", str(tmp_path))
    m = Model(name="m", namespace="n", kind="LLMInferenceService")
    assert cli._spawn_reconciler(m, 5)[1] and not cli._spawn_reconciler(m, 5)[1]
    assert len(held) == 1
    os.close(held[0])
    assert cli._spawn_reconciler(m, 5)[1]  # lock released once the reconciler exits


def test_proxy_precedence(monkeypatch, tmp_path):
    for var in ("MM_HTTPS_PROXY", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.delenv(var, raising=False)
    catalog = tmp_path / "models.yaml"
    catalog.write_text("models: []\n")
    monkeypatch.setenv("https_proxy", "http://std:1")
    assert Catalog(catalog).https_proxy == "http://std:1"
    monkeypatch.setenv("MM_HTTPS_PROXY", "http://mm:2")  # the Ansible install sets this one
    assert Catalog(catalog).https_proxy == "http://mm:2"
    catalog.write_text("network: {https_proxy: 'http://cfg:3'}\nmodels: []\n")
    assert Catalog(catalog).https_proxy == "http://cfg:3"


@pytest.mark.parametrize("global_cfg, inference_cfg, want", [
    ("", "", "keycloak"),
    ("", "auth_type: litellm_vkey", "litellm"),
    ("auth_provider: litellm", "", "litellm"),  # legacy key
    ("auth_type: jwt", "auth_type: litellm_vkey", "keycloak"),  # global_config.yaml wins, as in the installer
])
def test_auth_mode_from_configs(monkeypatch, tmp_path, global_cfg, inference_cfg, want):
    monkeypatch.delenv("MM_LITELLM_ENABLED", raising=False)
    for var, name, body in (("MM_GLOBAL_CONFIG", "global.yaml", global_cfg),
                            ("MM_INFERENCE_CONFIG", "inference.yaml", inference_cfg)):
        (tmp_path / name).write_text(body + "\n")
        monkeypatch.setenv(var, str(tmp_path / name))
    assert routing.auth_provider() == want


@pytest.mark.parametrize("global_cfg, inference_cfg, want", [
    ("base_domain_name: ex.test", "", ("inference.ex.test", "")),
    ("base_domain_name: ex.test\nauth_type: litellm_vkey", "", ("litellm.ex.test", "")),
    ("base_domain_name: ex.test\nrouting_mode: path", "", ("ex.test", "/inference")),
    ("base_domain_name: ex.test\nrouting_mode: path\nauth_type: litellm_vkey", "", ("ex.test", "/litellm")),
    ("base_domain_name: ex.test\nrouting_mode: path", "inference_path: /api/", ("ex.test", "/api")),
])
def test_endpoint_follows_routing_mode(monkeypatch, tmp_path, global_cfg, inference_cfg, want):
    for var in ("MM_LITELLM_ENABLED", "MM_INFERENCE_HOSTNAME", "MM_INFERENCE_PATH"):
        monkeypatch.delenv(var, raising=False)
    for var, name, body in (("MM_GLOBAL_CONFIG", "global.yaml", global_cfg),
                            ("MM_INFERENCE_CONFIG", "inference.yaml", inference_cfg)):
        (tmp_path / name).write_text(body + "\n")
        monkeypatch.setenv(var, str(tmp_path / name))
    assert (routing.inference_hostname(), routing.inference_path()) == want


def test_endpoint_env_overrides(monkeypatch, tmp_path):
    (tmp_path / "global.yaml").write_text("base_domain_name: ex.test\nrouting_mode: path\n")
    monkeypatch.setenv("MM_GLOBAL_CONFIG", str(tmp_path / "global.yaml"))
    monkeypatch.setenv("MM_INFERENCE_HOSTNAME", "api.ex.test")
    monkeypatch.setenv("MM_INFERENCE_PATH", "")  # set but empty: no path, as in subdomain mode
    assert (routing.inference_hostname(), routing.inference_path()) == ("api.ex.test", "")


@pytest.mark.parametrize("routing_mode, want", [
    ("subdomain", "https://litellm.ex.test/v1/chat/completions"),
    ("path", "https://ex.test/litellm/v1/chat/completions"),
])
def test_litellm_endpoint_is_unified(monkeypatch, tmp_path, capsys, routing_mode, want):
    for var in ("MM_INFERENCE_HOSTNAME", "MM_INFERENCE_PATH"):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / "global.yaml").write_text(f"base_domain_name: ex.test\nrouting_mode: {routing_mode}\n")
    monkeypatch.setenv("MM_GLOBAL_CONFIG", str(tmp_path / "global.yaml"))
    monkeypatch.setenv("MM_LITELLM_ENABLED", "true")
    monkeypatch.setenv("MM_AIGATEWAY_ENABLED", "false")
    monkeypatch.setattr(k8s, "out", lambda *a: "")
    routing.print_endpoints(Model(name="m", namespace="n", kind="LLMInferenceService"))
    assert f"Endpoint: {want}" in capsys.readouterr().out


def test_openshift_endpoint_ignores_path_mode(monkeypatch, tmp_path):
    for var in ("MM_LITELLM_ENABLED", "MM_INFERENCE_HOSTNAME", "MM_INFERENCE_PATH"):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / "global.yaml").write_text("base_domain_name: ex.test\nrouting_mode: path\n")
    monkeypatch.setenv("MM_GLOBAL_CONFIG", str(tmp_path / "global.yaml"))
    monkeypatch.setenv("MM_PLATFORM", "openshift")
    assert (routing.inference_hostname(), routing.inference_path()) == ("inference.ex.test", "")


@pytest.mark.parametrize("gateway_ip, want", [("", ""), ("10.0.0.9", "10.0.0.9")])
def test_openshift_gateway_ip_override(monkeypatch, capsys, gateway_ip, want):
    monkeypatch.setenv("MM_PLATFORM", "openshift")
    monkeypatch.setenv("MM_INFERENCE_HOSTNAME", "inference.ex.test")
    monkeypatch.setenv("MM_AIGATEWAY_ENABLED", "false")
    monkeypatch.setenv("MM_LITELLM_ENABLED", "false")
    monkeypatch.setenv("MM_GATEWAY_IP", gateway_ip)
    monkeypatch.setattr(routing, "_resolves", lambda host: True)
    routing.print_endpoints(Model(name="m", namespace="n", kind="LLMInferenceService"))
    out = capsys.readouterr().out
    assert (f"resolve inference.ex.test → {want}" in out) == bool(want)


def test_empty_host_does_not_resolve():
    assert not routing._resolves("") and not routing._resolves("  ")


@pytest.mark.parametrize("current, applied", [("", True), ("img:new", False), ("img:old", True)])
def test_runtime_installed_or_updated_on_image_change(monkeypatch, current, applied):
    calls = []
    monkeypatch.setattr(k8s, "out", lambda *a: current)
    monkeypatch.setattr(k8s, "apply", lambda manifest, **kw: calls.append(manifest))
    m = Model(name="m", namespace="n", kind="InferenceService", runtime="openvino", runtime_image="img:new",
              rt={"serving_runtime_manifest": "openvino/serving-runtime.yaml", "runtime_name": "openvino-runtime"})
    cli._install_runtime(m)
    assert bool(calls) == applied
    assert all("image: img:new" in c for c in calls)
@pytest.mark.parametrize("kind, want", [
    ("InferenceService", "http://m-predictor.n.svc.cluster.local:80/v1"),
    ("LLMInferenceService", "http://ai-gateway.envoy-gateway-system.svc.cluster.local/n/m/v1"),
])
def test_litellm_api_base(monkeypatch, kind, want):
    for var in ("MM_LITELLM_KSERVE_URL", "MM_INFERENCE_GATEWAY", "MM_INFERENCE_GATEWAY_NS", "MM_INFERENCE_GATEWAY_SVC_NS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(k8s, "out", lambda *a: "80" if "m-predictor" in a else "ai-gateway")
    assert routing._internal_url(Model(name="m", namespace="n", kind=kind)) == want


@pytest.mark.parametrize("kind, want", [
    ("InferenceService", "http://m-predictor.n.svc.cluster.local:80/v1"),
    ("LLMInferenceService", "http://gw.example.svc.cluster.local/n/m/v1"),
])
def test_litellm_api_base_override_is_gateway_only(monkeypatch, kind, want):
    monkeypatch.setenv("MM_LITELLM_KSERVE_URL", "http://gw.example.svc.cluster.local/")
    monkeypatch.setattr(k8s, "out", lambda *a: "80")
    assert routing._internal_url(Model(name="m", namespace="n", kind=kind)) == want


def test_litellm_called_directly_in_a_pod(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"ok": true}'

    class Opener:
        def open(self, req, timeout):
            seen.update(url=req.full_url, auth=req.get_header("Authorization"), method=req.get_method())
            return Resp()

    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setenv("MM_LITELLM_MASTER_KEY", "k")
    monkeypatch.setattr(routing.urllib.request, "build_opener", lambda *a: Opener())
    monkeypatch.setattr(k8s, "run", lambda *a, **kw: pytest.fail("no curl pod in a pod"))
    assert routing._litellm("POST", "http://litellm.litellm.svc:4000/model/new", {"a": 1}) == '{"ok": true}'
    assert seen == {"url": "http://litellm.litellm.svc:4000/model/new", "auth": "Bearer k", "method": "POST"}


FIXTURE_CATALOG = Path(__file__).resolve().parent / "fixtures" / "models.yaml"


def _llmisvc(name, ns, ready, reason=""):
    cond = {"type": "MainWorkloadReady", "status": ready, **({"reason": reason} if reason else {})}
    return {"metadata": {"name": name, "namespace": ns},
            "spec": {"replicas": 1, "template": {"containers": [{"image": "docker.io/vllm/vllm-openai-cpu:v0.24.0"}]}},
            "status": {"conditions": [cond]}}


@pytest.fixture
def cluster(monkeypatch):
    """Fake kubectl for list: answers `get <kind> ... -A` from `items`; `denied` kinds fail -A but not -n,
    `failing` kinds fail both."""
    state = {"items": {"LLMInferenceService": [], "InferenceService": []}, "denied": set(), "failing": set(),
             "down": False, "calls": []}

    def run(*args, stdin=None):
        state["calls"].append(args)
        if args[0] == "api-resources":
            return subprocess.CompletedProcess(args, 0, state.get("api_resources", ""), "")
        kind = args[1]
        if state["down"] or kind in state["failing"] or ("-A" in args and kind in state["denied"]):
            return subprocess.CompletedProcess(args, 1, "", "error")
        items = state["items"][kind]
        if "-n" in args:
            items = [i for i in items if i["metadata"]["namespace"] == args[args.index("-n") + 1]]
        return subprocess.CompletedProcess(args, 0, json.dumps({"items": items}), "")
    monkeypatch.setenv("MM_CONFIG", str(FIXTURE_CATALOG))
    monkeypatch.setenv("MM_PLATFORM", "")  # recorded, so the value list detects is undone after the test
    monkeypatch.delenv("MM_PLATFORM")
    monkeypatch.setattr(k8s, "run", run)
    return state


def test_list_marks_catalog_status_and_shows_other_deployments(cluster, capsys):
    cluster["items"]["LLMInferenceService"] = [
        _llmisvc("tmpl-inline", "ns1", "True"),
        _llmisvc("specdec", "ns1", "False", "MinimumReplicasUnavailable"),
        _llmisvc("my-alias", "ns1", "True"),             # --as / --hfid: not a catalog name
        _llmisvc("tmpl-path", "other", "True"),          # a catalog name, but not the catalog's namespace
    ]
    cli.list_models(cli.Options(), "")
    out = capsys.readouterr().out
    catalog, deployed = out.split("Deployed, not in the catalog")
    assert re.search(r"^tmpl-inline .* Ready$", catalog, re.M)
    assert re.search(r"^specdec .* Not ready \(MinimumReplicasUnavailable\)$", catalog, re.M)
    assert re.search(r"^tmpl-path .* not deployed$", catalog, re.M)
    assert re.search(r"^multi \(org/multi\) +llm +openvino +not deployed$", catalog, re.M)  # deploy's default server
    assert re.search(r"^my-alias +ns1 +LLMInferenceService +vllm-openai-cpu:v0\.24\.0 +1 +Ready$", deployed, re.M)
    assert re.search(r"^tmpl-path +other ", deployed, re.M)
    assert "tmpl-inline" not in deployed


def test_list_falls_back_to_the_catalog_namespace(cluster, capsys):
    cluster["denied"] = {"LLMInferenceService", "InferenceService"}
    cluster["items"]["LLMInferenceService"] = [_llmisvc("tmpl-inline", "ns1", "True"),
                                               _llmisvc("elsewhere", "other", "True")]
    cli.list_models(cli.Options(deployed_only=True), "")
    out = capsys.readouterr().out
    assert "tmpl-inline" in out and "elsewhere" not in out
    assert ("get", "LLMInferenceService", "-l", "app.kubernetes.io/managed-by=model-manager", "-o", "json",
            "-n", "ns1") in cluster["calls"]


@pytest.mark.parametrize("api_resources, want", [("clusterversions.config.openshift.io", r"Not ready \(NotAccepted\)"),
                                                  ("", "Ready")])
def test_list_status_on_openshift_covers_the_routes(cluster, capsys, api_resources, want):
    # Healthy pods, but the Gateway rejected the route: Ready=False. Kubernetes reads MainWorkloadReady.
    obj = _llmisvc("tmpl-inline", "ns1", "True")
    obj["status"]["conditions"].append({"type": "Ready", "status": "False", "reason": "NotAccepted"})
    cluster["items"]["LLMInferenceService"] = [obj]
    cluster["api_resources"] = api_resources
    cli.list_models(cli.Options(), "")
    assert re.search(rf"^tmpl-inline .* {want}$", capsys.readouterr().out, re.M)


def test_list_without_cluster_shows_unknown_status(cluster, capsys):
    cluster["down"] = True
    cli.list_models(cli.Options(), "")
    out, err = capsys.readouterr()
    assert "Cannot list deployed models" in err
    assert re.search(r"^tmpl-inline .* \?$", out, re.M) and "Deployed" not in out


def test_list_marks_unlisted_kind_unknown(cluster, capsys):
    cluster["failing"] = {"LLMInferenceService"}
    cli.list_models(cli.Options(), "")
    out, err = capsys.readouterr()
    assert "Cannot list LLMInferenceService" in err and "may be incomplete" in err
    assert re.search(r"^tmpl-inline .* \?$", out, re.M)                   # vllm: its kind was not listed
    assert re.search(r"^multi \(org/multi\) +llm +openvino +not deployed$", out, re.M)  # InferenceService was


def test_list_server_is_the_version_deploy_picks(cluster, tmp_path, monkeypatch, capsys):
    catalog = tmp_path / "models.yaml"
    catalog.write_text(
        "defaults: { namespace: ns1 }\n"
        "models:\n  - { name: eng-model, model_id: org/eng, runtime: eng }\n"
        "runtimes:\n  eng:\n    versions:\n"
        "      '0.9.1': { image: img/eng:0.9.1 }\n      '0.10.0': { image: img/eng:0.10.0 }\n")
    monkeypatch.setenv("MM_CONFIG", str(catalog))
    m = config.resolve(config.Catalog.load(), "eng-model", cli.Options())
    cli.list_models(cli.Options(catalog_only=True), "")
    assert re.search(rf"^eng-model \(org/eng\) +auto +eng {re.escape(m.version)}$", capsys.readouterr().out, re.M)


def test_list_catalog_needs_no_cluster(cluster, capsys):
    cli.list_models(cli.Options(catalog_only=True), "")
    out = capsys.readouterr().out
    assert cluster["calls"] == [] and "STATUS" not in out
    assert re.search(r"^nocat \(org/nocat\) +auto +vllm 0\.24\.0$", out, re.M)


@pytest.mark.parametrize("op, values, labels, want", [
    ("In", ["workload"], {"c": "workload"}, True), ("In", ["*"], {"c": "x"}, True), ("In", ["*"], {}, False),
    ("NotIn", ["workload"], {}, True), ("Equals", ["*"], {"c": "x"}, True), ("NotEqual", ["x"], {}, True),
    ("Matches", ["work*"], {"c": "workload"}, True), ("MatchesNot", ["work*"], {}, True),
    ("MatchesAny", ["a*", "w*"], {"c": "workload"}, True), ("MatchesNone", ["w*"], {"c": "workload"}, False),
    ("Exists", [], {"c": ""}, True), ("NotExist", [], {"c": "x"}, False), ("AlwaysTrue", [], {}, True),
    ("Bogus", ["x"], {"c": "x"}, False),
])
def test_label_matches_follows_nri_semantics(op, values, labels, want):
    assert nri_policy.label_matches(labels, [{"key": "pod/labels/c", "operator": op, "values": values}]) is want


@pytest.mark.parametrize("args, kubectl_on_path", [(["list", "--catalog"], False), (["list"], True)])
def test_list_works_offline_through_main(cluster, monkeypatch, capsys, args, kubectl_on_path):
    cluster["down"] = True
    for key in ("MM_PLATFORM", "MM_LLMISVC_API_VERSION"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda name: kubectl_on_path and f"/usr/bin/{name}")
    monkeypatch.setattr(sys, "argv", ["model-manager", *args])
    cli.main()
    assert re.search(r"^nocat \(org/nocat\) +auto +vllm 0\.24\.0", capsys.readouterr().out, re.M)


def test_list_flags_are_exclusive(cluster):
    with pytest.raises(SystemExit):
        cli.list_models(cli.Options(catalog_only=True, deployed_only=True), "")


@pytest.mark.parametrize("flag, warned", [("--hfid", False), ("--id", True)])
def test_hfid_and_deprecated_id_alias(capsys, flag, warned):
    o = cli.parse_args(["deploy", flag, "Qwen/Qwen3-0.6B"])
    assert o.id == "Qwen/Qwen3-0.6B" and o.positional == ["deploy"]
    assert ("--id is deprecated; use --hfid" in capsys.readouterr().err) is warned


LLMISVC = """apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: m
spec:
  annotations:
    balloon.balloons.resource-policy.nri.io/container.main: vllm-balloon-tp1
  model:
    uri: pvc://model-store/m
  template:
    containers:
      - name: main
        args:
          - "--served-model-name"
          - "served"
          - "--max-model-len=4096"
          - "--chat-template"
          - "/etc/chat-template/a b.jinja"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
"""


@pytest.mark.parametrize("value, expected", [("openshift", "openshift"), ("", "kubernetes")])
def test_platform_detect(monkeypatch, value, expected):
    from core import platform
    for key in ("MM_PLATFORM", "MM_LLMISVC_API_VERSION", "MM_INFERENCE_GATEWAY", "MM_AIGATEWAY_ENABLED"):
        monkeypatch.delenv(key, raising=False)
    answers = {"api-resources": "clusterversions.config.openshift.io\n" if value else "",
               "api-versions": "serving.kserve.io/v1alpha1\n"}
    monkeypatch.setattr(k8s, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, answers.get(a[0], ""), ""))
    platform.detect()
    assert os.environ["MM_PLATFORM"] == expected and os.environ["MM_LLMISVC_API_VERSION"] == "v1alpha1"
    assert (os.environ.get("MM_INFERENCE_GATEWAY") == "openshift-ai-inference") == bool(value)


@pytest.mark.parametrize("value, expected", [
    ("", (("openshift-ai-inference", "openshift-ingress"), False)),
    ("x", (("x", "x"), True)),
])
def test_openshift_gateway_overrides(monkeypatch, value, expected):
    # An empty override falls back to the OpenShift Gateway and keeps the AI gateway off; a set one is kept.
    from core import platform
    monkeypatch.setenv("MM_PLATFORM", "openshift")
    monkeypatch.setenv("MM_LLMISVC_API_VERSION", "v1alpha1")
    for key in ("MM_INFERENCE_GATEWAY", "MM_INFERENCE_GATEWAY_NS", "MM_AIGATEWAY_ENABLED"):
        monkeypatch.setenv(key, value)
    platform.detect()
    assert (config.gateway_ref(), routing.gateway_enabled()) == expected


@pytest.mark.parametrize("stdout, aborts", [("", True), ("clusterversions.config.openshift.io\n", False)])
def test_platform_detect_failure(monkeypatch, stdout, aborts):
    # Forbidden prints nothing: abort rather than assume Kubernetes. A stale aggregated API
    # fails discovery but still prints the list.
    from core import platform
    for key in ("MM_PLATFORM", "MM_LLMISVC_API_VERSION", "MM_INFERENCE_GATEWAY", "MM_AIGATEWAY_ENABLED"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(k8s, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 1, stdout, "Forbidden"))
    if aborts:
        with pytest.raises(SystemExit):
            platform.detect()
    else:
        platform.detect()
        assert os.environ["MM_PLATFORM"] == "openshift"


def test_openshift_llmisvc(monkeypatch):
    # OpenShift AI 3.3: v1alpha1, vLLM started by its template script, no pinned UIDs.
    import yaml
    from core import platform
    monkeypatch.setenv("MM_PLATFORM", "openshift")
    monkeypatch.setenv("MM_LLMISVC_API_VERSION", "v1alpha1")
    doc = yaml.safe_load(platform.adapt_llmisvc(platform.adapt(LLMISVC)))
    main = doc["spec"]["template"]["containers"][0]
    assert doc["apiVersion"] == "serving.kserve.io/v1alpha1" and "annotations" not in doc["spec"]
    assert doc["metadata"]["annotations"] == {"balloon.balloons.resource-policy.nri.io/container.main": "vllm-balloon-tp1"}
    assert doc["spec"]["model"] == {"uri": "pvc://model-store/m", "name": "served"}
    assert "args" not in main and main["env"] == [
        {"name": "VLLM_ADDITIONAL_ARGS", "value": "--max-model-len=4096 --chat-template '/etc/chat-template/a b.jinja'"}]
    assert main["readinessProbe"]["httpGet"]["scheme"] == "HTTPS"
    assert main["securityContext"] == {"runAsNonRoot": True}


def test_openshift_llmisvc_keeps_given_vllm_args(monkeypatch):
    # A VLLM_ADDITIONAL_ARGS from the catalog or --env stays, after the translated args.
    import yaml
    from core import platform
    monkeypatch.setenv("MM_PLATFORM", "openshift")
    monkeypatch.setenv("MM_LLMISVC_API_VERSION", "v1alpha1")
    doc = yaml.safe_load(platform.adapt(LLMISVC))
    doc["spec"]["template"]["containers"][0]["env"] = [
        {"name": "VLLM_ADDITIONAL_ARGS", "value": "--enable-auto-tool-choice --tool-call-parser hermes"}]
    main = yaml.safe_load(platform.adapt_llmisvc(yaml.safe_dump(doc)))["spec"]["template"]["containers"][0]
    assert main["env"] == [{"name": "VLLM_ADDITIONAL_ARGS", "value": "--max-model-len=4096 --chat-template "
                            "'/etc/chat-template/a b.jinja' --enable-auto-tool-choice --tool-call-parser hermes"}]


def test_kubernetes_llmisvc_unchanged(monkeypatch):
    from core import platform
    monkeypatch.setenv("MM_PLATFORM", "kubernetes")
    monkeypatch.setenv("MM_LLMISVC_API_VERSION", "v1alpha2")
    assert platform.adapt_llmisvc(platform.adapt(LLMISVC)) == LLMISVC


def test_openshift_skips_pool_status(monkeypatch, kubectl):
    monkeypatch.setenv("MM_PLATFORM", "openshift")
    routing.patch_pool_status(Model(name="m", namespace="ns"))
    assert kubectl == []


def test_balloon_cpu_counts_label_selected_pods(monkeypatch):
    # OpenShift AI's v1alpha1 models land in their balloon by label, without the annotation:
    # the capacity check must count them like annotated pods, and nothing else.
    from core import nri_policy

    def pod(name, cpu, labels=None, annotations=None, phase="Running"):
        return {"metadata": {"name": name, "labels": labels or {}, "annotations": annotations or {}},
                "status": {"phase": phase},
                "spec": {"containers": [{"resources": {"requests": {"cpu": cpu}}}]}}
    policies = {"items": [{"spec": {"balloonTypes": [
        {"name": "vllm-balloon-tp1", "matchExpressions": [
            {"key": "pod/labels/kserve.io/component", "operator": "In", "values": ["workload"]}]},
        {"name": "default"}]}}]}
    pods = {"items": [
        pod("annotated", "4", annotations={"balloon.balloons.resource-policy.nri.io/container.main": "vllm-balloon"}),
        pod("by-label", "8", labels={"kserve.io/component": "workload"}),
        pod("by-label-done", "16", labels={"kserve.io/component": "workload"}, phase="Succeeded"),
        pod("other", "2000m", labels={"app": "x"})]}
    monkeypatch.setattr(nri_policy, "_kubectl_json",
                        lambda *a: policies if a[1] == nri_policy._BALLOONSPOLICY_CRD else pods)
    assert nri_policy._balloon_pods_cpu_on_node("n1") == 12


@pytest.mark.parametrize("platform_name, want", [("kubernetes", "True"), ("openshift", "False")])
def test_workload_ready_waits_for_routes_on_openshift(monkeypatch, platform_name, want):
    monkeypatch.setenv("MM_PLATFORM", platform_name)
    conditions = [{"type": "MainWorkloadReady", "status": "True"}, {"type": "Ready", "status": "False"}]
    monkeypatch.setattr(k8s, "get_json", lambda *a: {"status": {"conditions": conditions}})
    assert cli.workload_ready(Model(name="m", namespace="n", kind="LLMInferenceService")) == want


@pytest.mark.parametrize("host_network, want", [("true", "10.0.0.5"), ("false", "")])
def test_router_ip_node_fallback_needs_host_network(monkeypatch, host_network, want):
    from core import platform
    monkeypatch.delenv("MM_GATEWAY_IP", raising=False)
    answers = {"deploy": host_network, "pods": "10.0.0.5"}
    monkeypatch.setattr(k8s, "out", lambda *a: answers.get(a[1], ""))
    assert platform.router_ip() == want


# ── Autoscaling (KEDA) ────────────────────────────────────────────────────────

def _render(monkeypatch, *argv, catalog=None):
    import yaml
    from core import renderer
    monkeypatch.setenv("MM_PLATFORM", "kubernetes")
    monkeypatch.setenv("MM_LLMISVC_API_VERSION", "v1alpha2")
    o = cli.parse_args(["deploy", *argv])
    cat = Catalog(catalog or cli.MM_DIR / "models.yaml")
    m = config.resolve(cat, o.positional[1] if len(o.positional) > 1 else "", o)
    config.resolve_autoscaling(m, o)
    return m, yaml.safe_load(renderer.manifest(m, cat, o, {}, m.cpu, []))


def test_autoscaling_renders_keda_scaling_instead_of_replicas(monkeypatch):
    monkeypatch.setenv("MM_PROMETHEUS_URL", "http://prom:9090")
    m, doc = _render(monkeypatch, "qwen3-0-6b", "--max-replicas", "3", "--scale-target", "4")
    assert m.autoscaled and m.replicas == "1"
    assert "replicas" not in doc["spec"]  # the CRD rejects replicas together with scaling
    assert doc["spec"]["scaling"] == {"minReplicas": 1, "maxReplicas": 3, "keda": {"pollingInterval": 15, "triggers": [
        {"type": "prometheus", "metadata": {
            "serverAddress": "http://prom:9090", "threshold": "4",
            "query": 'sum(vllm:num_requests_waiting{namespace="llm-inference",model="qwen3-0-6b"})'}}]}}


def test_autoscaling_from_catalog_and_fixed_replicas_override(monkeypatch, tmp_path):
    catalog = tmp_path / "models.yaml"
    catalog.write_text((cli.MM_DIR / "models.yaml").read_text().replace(
        "  - name: qwen3-0-6b ",
        "  - name: qwen3-0-6b\n    autoscaling: {min_replicas: 2, max_replicas: 4, scale_target: 5}\n    x: ", 1))
    m, doc = _render(monkeypatch, "qwen3-0-6b", catalog=catalog)
    assert (doc["spec"]["scaling"]["minReplicas"], doc["spec"]["scaling"]["maxReplicas"]) == (2, 4)
    assert doc["spec"]["scaling"]["keda"]["triggers"][0]["metadata"]["threshold"] == "5"
    m, doc = _render(monkeypatch, "qwen3-0-6b", "--replicas", "2", catalog=catalog)
    assert not m.autoscaled and doc["spec"]["replicas"] == 2 and "scaling" not in doc["spec"]


def test_equal_min_and_max_is_a_fixed_count(monkeypatch):
    m, doc = _render(monkeypatch, "qwen3-0-6b", "--min-replicas", "2", "--max-replicas", "2")
    assert not m.autoscaled and doc["spec"]["replicas"] == 2 and "scaling" not in doc["spec"]


@pytest.mark.parametrize("argv", [
    ["--replicas", "2", "--max-replicas", "3"],
    ["--replicas", "2", "--scale-target", "4"],
    ["--min-replicas", "4", "--max-replicas", "3"],
    ["--min-replicas", "2"],
    ["--scale-target", "4"],  # a threshold without a maximum would be ignored
    ["--max-replicas", "0"],
])
def test_bad_autoscaling_rejected(monkeypatch, argv):
    with pytest.raises(SystemExit):
        _render(monkeypatch, "qwen3-0-6b", *argv)


def test_ovms_autoscaling_renders_keda_on_the_predictor(monkeypatch):
    monkeypatch.setenv("MM_PROMETHEUS_URL", "http://prom:9090")
    m, doc = _render(monkeypatch, "qwen3-5-2b-ovms", "--max-replicas", "3")
    assert doc["kind"] == "InferenceService" and m.scale_target == "4"  # the runtime's default
    assert doc["metadata"]["annotations"] == {"serving.kserve.io/autoscalerClass": "keda"}
    assert doc["metadata"]["labels"]["inference-engine"] == "ovms"  # the ovms PodMonitor selects it
    predictor = doc["spec"]["predictor"]
    assert (predictor["minReplicas"], predictor["maxReplicas"]) == (1, 3)
    assert predictor["autoScaling"] == {"metrics": [{"type": "External", "external": {
        "metric": {"backend": "prometheus", "serverAddress": "http://prom:9090",
                   "query": 'sum(ovms_current_graphs{namespace="llm-inference",model="qwen3-5-2b-ovms"})'},
        "target": {"type": "Value", "value": "4"}}}]}


def test_ovms_fixed_replicas_unchanged(monkeypatch):
    _, doc = _render(monkeypatch, "qwen3-5-2b-ovms", "--replicas", "2")
    assert "annotations" not in doc["metadata"] and "autoScaling" not in doc["spec"]["predictor"]
    assert (doc["spec"]["predictor"]["minReplicas"], doc["spec"]["predictor"]["maxReplicas"]) == (2, 2)


def test_query_uses_the_deployed_name(monkeypatch):
    _, doc = _render(monkeypatch, "qwen3-0-6b", "--as", "chat", "--max-replicas", "2")
    assert 'model="chat"' in doc["spec"]["scaling"]["keda"]["triggers"][0]["metadata"]["query"]


def test_runtime_without_a_signal_is_rejected(monkeypatch, tmp_path):
    catalog = tmp_path / "models.yaml"
    catalog.write_text((cli.MM_DIR / "models.yaml").read_text().replace(
        "  openvino:\n", "  openvino:\n    autoscaling: {query: \"\"}\n", 1))
    with pytest.raises(SystemExit):
        _render(monkeypatch, "qwen3-5-2b-ovms", "--max-replicas", "3", catalog=catalog)


def test_namespace_must_be_a_dns_label(monkeypatch, tmp_path):
    catalog = tmp_path / "models.yaml"
    catalog.write_text((cli.MM_DIR / "models.yaml").read_text().replace(
        "  namespace: llm-inference ", '  namespace: \'x"}) or vector(1\' ', 1))
    monkeypatch.setenv("MM_PLATFORM", "kubernetes")
    m = config.resolve(Catalog(catalog), "qwen3-0-6b", cli.parse_args(["deploy", "qwen3-0-6b", "--max-replicas", "3"]))
    with pytest.raises(SystemExit):  # it goes into PromQL
        m.check()


def test_autoscaling_not_on_openshift(monkeypatch):
    monkeypatch.setenv("MM_PLATFORM", "openshift")
    o = cli.parse_args(["deploy", "qwen3-0-6b", "--max-replicas", "3"])
    m = config.resolve(Catalog(cli.MM_DIR / "models.yaml"), "qwen3-0-6b", o)  # undeploy resolves too: no checks
    with pytest.raises(SystemExit):
        config.resolve_autoscaling(m, o)


@pytest.mark.parametrize("crd, explain, aborts", [(False, True, True), (True, False, True), (True, True, False)])
def test_autoscaling_needs_keda_and_a_kserve_with_keda_scaling(monkeypatch, crd, explain, aborts):
    # KServe 0.18's LLMInferenceService has spec.scaling.wva only: the apply would fail on spec.scaling.keda.
    def run(*args, stdin=None):
        ok = crd if args[0] == "get" else explain
        return subprocess.CompletedProcess(args, 0 if ok else 1, "", "" if ok else "error")
    monkeypatch.setattr(k8s, "run", run)
    m = Model(name="m", namespace="n", kind="LLMInferenceService")
    if aborts:
        with pytest.raises(SystemExit):
            cli._require_autoscaling(m)
    else:
        cli._require_autoscaling(m)


def test_list_shows_the_autoscaling_range():
    assert cli._replicas({"scaling": {"minReplicas": 1, "maxReplicas": 3}}, {}) == "1-3"
    assert cli._replicas({"replicas": 2}, {}) == "2"
    assert cli._replicas({}, {"minReplicas": 1, "maxReplicas": 3}) == "1-3"  # autoscaled InferenceService
    assert cli._replicas({}, {"minReplicas": 2, "maxReplicas": 2}) == "2"


@pytest.fixture
def deploy_env(monkeypatch):
    """deploy() against a cluster where every kubectl call fails (no KEDA, nothing deployed)."""
    calls = []
    for k, v in {"MM_PLATFORM": "kubernetes", "MM_LLMISVC_API_VERSION": "v1alpha2", "MM_CPU_POLICY": "best-effort",
                 "MM_CONFIG": str(cli.MM_DIR / "models.yaml")}.items():
        monkeypatch.setenv(k, v)

    def run(*args, stdin=None):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1, "", "NotFound")
    monkeypatch.setattr(k8s, "run", run)
    return calls


def test_dry_run_needs_no_keda(deploy_env, capsys):
    cli.deploy(cli.parse_args(["deploy", "qwen3-0-6b", "--max-replicas", "3", "--dry-run"]), "qwen3-0-6b")
    assert "scaling:" in capsys.readouterr().out
    assert not any(a[:2] == ("get", "crd") for a in deploy_env)


def test_missing_keda_aborts_before_any_change(deploy_env):
    with pytest.raises(SystemExit):
        cli.deploy(cli.parse_args(["deploy", "qwen3-0-6b", "--max-replicas", "3"]), "qwen3-0-6b")
    assert ("get", "crd", "scaledobjects.keda.sh") in deploy_env
    assert not any(a[0] in ("apply", "create", "delete") for a in deploy_env)


@pytest.mark.parametrize("fits_max, warned", [(True, False), (False, True)])
def test_max_replicas_only_warns(capsys, fits_max, warned):
    class Policy:
        def replica_shortfall(self, m, tp):
            return "" if fits_max or m.replicas != "3" else "too big"
    m = Model(name="m", namespace="n", replicas="1", min_replicas="1", max_replicas="3")
    policy.warn_max_unplaceable(Policy(), m, 1)
    assert m.replicas == "1"  # checked on a copy
    assert ("do not all fit" in capsys.readouterr().err) == warned


@pytest.mark.parametrize("node_cpus, tp, warned", [
    ([8, 8], 1, True),     # 24 CPUs needed, 16 allocatable
    ([12, 12], 1, True),   # 24 = 24 in total, but each node fits one 8-CPU replica
    ([16, 8], 1, False),   # 2 + 1 slots
    ([24], 2, False),      # TP splits the 8 CPUs a pod requests, it does not multiply them
])
def test_kubelet_static_max_shortfall_counts_slots_per_node(monkeypatch, capsys, node_cpus, tp, warned):
    nodes = {"items": [{"status": {"allocatable": {"cpu": str(c)}}, "spec": {}} for c in node_cpus]}
    monkeypatch.setattr(k8s, "get_json", lambda *args: nodes)
    m = Model(name="m", namespace="n", cpu="8", replicas="1", min_replicas="1", max_replicas="3")
    policy.warn_max_unplaceable(policy.KubeletStatic(), m, tp)
    assert ("3 replicas x 8 CPUs: the schedulable nodes fit" in capsys.readouterr().err) == warned


def test_nri_max_shortfall_reported_in_warn_mode(monkeypatch, capsys):
    # MM_NRI_PREFLIGHT=warn makes the gate pass; the max check still reports the errors.
    monkeypatch.setattr(nri_policy, "preflight", lambda *a, **k: {"ok": True, "errors": ["no room"]})
    m = Model(name="m", namespace="n", cpu="8", replicas="1", min_replicas="1", max_replicas="3")
    assert policy.NriBalloons().preflight(m, 1) == []
    policy.warn_max_unplaceable(policy.NriBalloons(), m, 1)
    assert "do not all fit now (no room)" in capsys.readouterr().err


@pytest.mark.parametrize("autoscaled, kind", [(True, "hpa"), (False, "scaledobject.keda.sh")])
@pytest.mark.parametrize("owner, removed", [("m", True), ("other", False)])
def test_stale_autoscaler_of_the_other_class_removed(monkeypatch, autoscaled, kind, owner, removed):
    # kserve#5196: KServe keeps the old class's autoscaler, and KEDA then rejects the ScaledObject.
    calls = []

    def run(*args, stdin=None):
        calls.append(args)
        obj = {"metadata": {"ownerReferences": [{"kind": "InferenceService", "name": owner}]}}
        return subprocess.CompletedProcess(args, 0, json.dumps(obj) if args[0] == "get" else "", "")
    monkeypatch.setattr(k8s, "run", run)
    m = Model(name="m", namespace="ns", kind="InferenceService",
              **({"min_replicas": "1", "max_replicas": "3"} if autoscaled else {}))
    cli._drop_stale_autoscaler(m)
    assert calls[0][:3] == ("get", kind, "m-predictor")
    assert (("delete", kind, "m-predictor", "-n", "ns", "--ignore-not-found") in calls) == removed


@pytest.mark.parametrize("kind, autoscaled, changed, want", [
    # To KEDA: the HPA goes first; KServe does not watch HPAs, so the apply's reconcile finds none.
    ("InferenceService", True, True, ["get hpa", "delete hpa", "apply"]),
    # To fixed: the ScaledObject goes after; KServe watches them and would recreate one under class keda.
    ("InferenceService", False, True, ["apply", "get scaledobject.keda.sh", "delete scaledobject.keda.sh"]),
    # Unchanged spec: no apply, the cleanup still repairs a model already stuck.
    ("InferenceService", True, False, ["get hpa", "delete hpa"]),
    ("InferenceService", False, False, ["get scaledobject.keda.sh", "delete scaledobject.keda.sh"]),
    ("LLMInferenceService", True, True, ["apply"]),  # KServe hands over by itself
])
def test_autoscaler_handover_order(monkeypatch, kind, autoscaled, changed, want):
    calls = []

    def run(*args, stdin=None):
        calls.append(" ".join(args[:2]))
        owned = {"metadata": {"ownerReferences": [{"kind": "InferenceService", "name": "m"}]}}
        return subprocess.CompletedProcess(args, 0, json.dumps(owned) if args[0] == "get" else "", "")
    monkeypatch.setattr(k8s, "run", run)
    m = Model(name="m", namespace="ns", kind=kind, **({"min_replicas": "1", "max_replicas": "3"} if autoscaled else {}))
    cli._apply_manifest(m, "manifest", changed)
    assert [c.replace("apply --server-side", "apply") for c in calls] == want


def test_dry_run_changes_no_autoscaler(deploy_env, capsys):
    cli.deploy(cli.parse_args(["deploy", "qwen3-5-2b-ovms", "--max-replicas", "3", "--dry-run"]), "qwen3-5-2b-ovms")
    assert "autoScaling:" in capsys.readouterr().out
    assert not any(a[0] in ("apply", "delete") for a in deploy_env)


@pytest.mark.parametrize("model", ["qwen3-0-6b", "qwen3-5-2b-ovms"])
def test_runtime_default_target_needs_no_max(monkeypatch, model):
    # The runtime's default scale_target applies only once autoscaling is on; a plain deploy stays fixed.
    m, _ = _render(monkeypatch, model)
    assert not m.autoscaled and m.replicas == "1" and m.scale_target == ""
