"""Unit tests for behaviour the parity suite cannot observe, mostly fixes over the legacy CLI."""
from __future__ import annotations

import json
import os
import re
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import cli, config, download, k8s, nri_policy, policy, renderer, routing  # noqa: E402
from core.config import Catalog, Model  # noqa: E402
from core.policy import whole_cpus  # noqa: E402
from core.renderer import dedupe_args, engine_args  # noqa: E402


@pytest.fixture
def kubectl(monkeypatch):
    """Record kubectl calls as (argv, stdin); every call succeeds with empty output (a Job reads as complete)."""
    calls = []

    def run(*args, stdin=None):
        calls.append((args, stdin))
        out = '{"status": {"succeeded": 1}}' if args[:2] == ("get", "job") else ""
        return subprocess.CompletedProcess(args, 0, out, "")
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
    assert whole_cpus(cpu) == whole


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
def test_runtime_installed_or_updated_on_image_change(monkeypatch, tmp_path, current, applied):
    calls = []
    (tmp_path / "isvc").mkdir()
    (tmp_path / "isvc" / "serving-runtime.yaml").write_text(
        "apiVersion: serving.kserve.io/v1alpha1\nkind: ClusterServingRuntime\nmetadata:\n  name: isvc-runtime\n"
        "spec:\n  containers:\n    - name: kserve-container\n      image: {{RUNTIME_IMAGE}}\n")
    monkeypatch.setenv("MM_RUNTIMES", str(tmp_path))
    monkeypatch.setattr(k8s, "out", lambda *a: current)
    monkeypatch.setattr(k8s, "apply", lambda manifest, **kw: calls.append(manifest))
    m = Model(name="m", namespace="n", kind="InferenceService", runtime="isvc", runtime_image="img:new",
              rt={"runtime_name": "isvc-runtime"})
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


def test_list_marks_unlisted_kind_unknown(cluster, capsys, monkeypatch, tmp_path):
    catalog = tmp_path / "models.yaml"
    catalog.write_text(FIXTURE_CATALOG.read_text().replace(
        "  - name: multi\n", "  - name: multi\n    kind: InferenceService\n", 1))
    monkeypatch.setenv("MM_CONFIG", str(catalog))
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


@pytest.mark.parametrize("kind, env", [("InferenceService", "kubernetes"), ("", "openshift")])
def test_autoscaling_needs_llmisvc_on_kubernetes(monkeypatch, tmp_path, kind, env):
    catalog = tmp_path / "models.yaml"
    text = (cli.MM_DIR / "models.yaml").read_text()
    catalog.write_text(re.sub(r"^(  - name: qwen3-0-6b\b.*\n)", r"\g<1>" + (f"    kind: {kind}\n" if kind else ""),
                              text, count=1, flags=re.MULTILINE))
    monkeypatch.setenv("MM_PLATFORM", env)
    o = cli.parse_args(["deploy", "qwen3-0-6b", "--max-replicas", "3"])
    m = config.resolve(Catalog(catalog), "qwen3-0-6b", o)  # undeploy resolves too: no checks here
    with pytest.raises(SystemExit):
        config.resolve_autoscaling(m, o)


@pytest.mark.parametrize("crd, explain, aborts", [(False, True, True), (True, False, True), (True, True, False)])
def test_autoscaling_needs_keda_and_a_kserve_with_keda_scaling(monkeypatch, crd, explain, aborts):
    # KServe 0.18's LLMInferenceService has spec.scaling.wva only: the apply would fail on spec.scaling.keda.
    monkeypatch.setenv("MM_LLMISVC_API_VERSION", "v1alpha1")
    calls = []

    def run(*args, stdin=None):
        calls.append(args)
        ok = crd if args[0] == "get" else explain
        return subprocess.CompletedProcess(args, 0 if ok else 1, "", "" if ok else "error")
    monkeypatch.setattr(k8s, "run", run)
    m = Model(name="m", namespace="n", kind="LLMInferenceService")
    if aborts:
        with pytest.raises(SystemExit):
            cli._require_autoscaling(m)
    else:
        cli._require_autoscaling(m)
    explained = [c for c in calls if c[0] == "explain"]
    assert not crd or explained[0][:3] == ("explain", "--api-version", "serving.kserve.io/v1alpha1")


def test_list_shows_the_autoscaling_range():
    assert cli._replicas({"scaling": {"minReplicas": 1, "maxReplicas": 3}}, {}) == "1-3"
    assert cli._replicas({"replicas": 2}, {}) == "2"


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


def test_running_replicas_counts_pods_holding_cpus(monkeypatch):
    # As NRI counts cores in use: on a node and Running/Pending; unscheduled and finished pods hold none.
    pods = {"items": [{"spec": {"nodeName": "w1"}, "status": {"phase": "Running"}},
                      {"spec": {"nodeName": "w1"}, "status": {"phase": "Pending"}},
                      {"spec": {}, "status": {"phase": "Pending"}},
                      {"spec": {"nodeName": "w1"}, "status": {"phase": "Failed"}}]}
    asked = []
    monkeypatch.setattr(k8s, "get_json", lambda *a: asked.append(a) or pods)
    assert cli._running_replicas(Model(name="m", namespace="n", kind="LLMInferenceService")) == 2
    assert asked[0][-1] == "app.kubernetes.io/name=m,kserve.io/component=workload"
    monkeypatch.setattr(k8s, "get_json", lambda *a: None)
    assert cli._running_replicas(Model(name="m", namespace="n")) == 0


def test_no_schedulable_node_is_a_shortfall():
    assert "the schedulable nodes fit 0" in policy._replica_shortfall([], 8, 3)  # every node tainted NoSchedule
    assert policy._replica_shortfall(None, 8, 3) == ""  # nodes could not be read


def _autoscaled(argv):
    o = cli.parse_args(["deploy", "qwen3-0-6b", *argv])
    cat = Catalog(cli.MM_DIR / "models.yaml")
    m = config.resolve(cat, "qwen3-0-6b", o)
    config.resolve_autoscaling(m, o)
    return m, lambda model: renderer.manifest(model, cat, o, {}, model.cpu, [])


LIVE_1_3 = {"scaling": {"minReplicas": 1, "maxReplicas": 3, "keda": {"triggers": [{"metadata": {"threshold": "2"}}]}}}


@pytest.mark.parametrize("spec, want", [
    ({"replicas": 2}, ("2", "", "", "")),
    (LIVE_1_3, ("1", "1", "3", "2")),
    ({"scaling": {"minReplicas": 1, "maxReplicas": 3, "keda": {"triggers": []}}}, ("1", "1", "3", "")),
    ({}, ("1", "", "", "")),
])
def test_live_scaling_reads_back_the_rendered_range(monkeypatch, spec, want):
    monkeypatch.setenv("MM_PLATFORM", "kubernetes")
    m, _ = _autoscaled(["--max-replicas", "4"])
    live = renderer.live_scaling(m, spec)
    assert (live.replicas, live.min_replicas, live.max_replicas, live.scale_target) == want


def test_live_scaling_of_an_inference_service():
    m = Model(name="m", namespace="n", kind="InferenceService", replicas="1")
    assert renderer.live_scaling(m, {"predictor": {"minReplicas": 2}}).replicas == "2"
    assert renderer.live_scaling(m, {}).replicas == "1"


@pytest.mark.parametrize("diff_rc, want", [(0, True), (1, False)])
def test_only_scaling_changed(monkeypatch, diff_rc, want):
    # Live 1-3 with threshold 2, redeployed as 1-4: m is re-rendered with the live range and diffed.
    monkeypatch.setenv("MM_PLATFORM", "kubernetes")
    m, render = _autoscaled(["--max-replicas", "4"])
    diffed = []

    def run(*args, stdin=None):
        if args[0] == "get":
            return subprocess.CompletedProcess(args, 0, json.dumps({"spec": LIVE_1_3}), "")
        diffed.append(stdin)
        return subprocess.CompletedProcess(args, diff_rc, "", "")
    monkeypatch.setattr(k8s, "run", run)
    assert cli._only_scaling_changed(m, render) == want
    assert "maxReplicas: 3" in diffed[0] and re.search(r"threshold: .?2.?\n", diffed[0])
    assert m.max_replicas == "4"  # compared on a copy


def test_only_scaling_changed_needs_a_live_spec(monkeypatch):
    monkeypatch.setattr(k8s, "get_json", lambda *a: None)
    assert cli._only_scaling_changed(Model(name="m", namespace="n"), lambda model: "") is False


# ── OVMS as an LLMInferenceService (the runtime's own container) ─────────────

def _main(doc):
    return doc["spec"]["template"]["containers"][0]


def test_ovms_renders_an_llmisvc_with_its_own_container(monkeypatch):
    m, doc = _render(monkeypatch, "qwen3-5-2b-ovms")
    main = _main(doc)
    assert doc["kind"] == "LLMInferenceService" and m.routing == "epp" and doc["spec"]["replicas"] == 1
    assert doc["spec"]["router"]["scheduler"] == {}  # KServe's scheduler routes each request
    assert main["image"] == "openvino/model_server:2026.4.1" and main["command"] == ["/ovms/bin/ovms"]
    assert main["args"][:7] == ["--config_path", "/mnt/models/config_all.json", "--rest_port", "8000", "--port",
                                "9000", "--metrics_enable"]
    assert "--served-model-name" not in main["args"]  # a vLLM flag
    # Ready once the graph is loaded, not when the REST port opens.
    for probe in ("startupProbe", "readinessProbe"):
        assert main[probe]["httpGet"] == {"path": "/v2/models/qwen3-5-2b-ovms/ready", "port": 8000}
    assert main["livenessProbe"]["httpGet"]["path"] == "/v2/health/live"
    assert (main["securityContext"]["runAsUser"], main["securityContext"]["runAsGroup"]) == (5000, 5000)


def test_vllm_keeps_kserves_container(monkeypatch):
    _, doc = _render(monkeypatch, "qwen3-0-6b")
    main = _main(doc)
    assert "command" not in main and "--served-model-name" in main["args"]
    assert {main[p]["httpGet"]["path"] for p in ("startupProbe", "readinessProbe", "livenessProbe")} == {"/health"}
    assert main["securityContext"]["runAsUser"] == 1000


def test_ovms_autoscaling_uses_its_signal(monkeypatch):
    monkeypatch.setenv("MM_PROMETHEUS_URL", "http://prom:9090")
    m, doc = _render(monkeypatch, "qwen3-5-2b-ovms", "--max-replicas", "3")
    assert "replicas" not in doc["spec"] and m.scale_target == "4"  # the runtime's default
    assert doc["spec"]["scaling"]["keda"]["triggers"][0]["metadata"] == {
        "serverAddress": "http://prom:9090", "threshold": "4",
        "query": 'sum(ovms_current_graphs{namespace="llm-inference",model="qwen3-5-2b-ovms"})'}


def test_servers_sorted_default_is_an_llmisvc():
    # was the fx-servers-sorted-default parity case: no default_server picks the first server by name
    m = config.resolve(Catalog(FIXTURE_CATALOG), "multi", cli.Options())
    assert (m.runtime, m.kind) == ("openvino", "LLMInferenceService")


def test_files_belong_to_the_serving_user():
    cat = Catalog(cli.MM_DIR / "models.yaml")
    assert config.resolve(cat, "qwen3-5-2b-ovms", cli.Options()).run_as_user == "5000"
    assert config.resolve(cat, "qwen3-0-6b", cli.Options()).run_as_user == "1000"


def test_vllm_only_features_rejected_for_ovms(monkeypatch, tmp_path):
    catalog = tmp_path / "models.yaml"
    catalog.write_text((cli.MM_DIR / "models.yaml").read_text().replace(
        "  - name: qwen3-5-2b-ovms\n", "  - name: qwen3-5-2b-ovms\n    chat_template: '{{ messages }}'\n", 1))
    with pytest.raises(SystemExit):
        _render(monkeypatch, "qwen3-5-2b-ovms", catalog=catalog)


def test_ovms_not_on_openshift(monkeypatch):
    from core import renderer
    o = cli.parse_args(["deploy", "qwen3-5-2b-ovms"])
    cat = Catalog(cli.MM_DIR / "models.yaml")
    m = config.resolve(cat, "qwen3-5-2b-ovms", o)
    monkeypatch.setenv("MM_PLATFORM", "openshift")
    with pytest.raises(SystemExit):  # OpenShift AI's template starts vLLM itself
        renderer.manifest(m, cat, o, {}, m.cpu, [])


@pytest.mark.parametrize("owner, retired", [("model-manager", True), ("someone-else", False), ("", False)])
def test_owned_inference_service_retired(monkeypatch, owner, retired):
    # An upgrade leaves an OVMS model's InferenceService; the LLMInferenceService replaces it.
    monkeypatch.setenv("MM_AIGATEWAY_ENABLED", "true")
    monkeypatch.setenv("MM_AI_GATEWAY_PROVIDER", "envoy")
    calls = []

    def run(*args, stdin=None):
        calls.append(" ".join(args[:3]))
        if args[:2] == ("get", "InferenceService"):
            return subprocess.CompletedProcess(args, 0 if owner else 1, owner, "")
        found = args[0] == "delete" or args[:3] in (("get", "aigatewayroute", "aigw-m"), ("get", "aiservicebackend", "m"))
        return subprocess.CompletedProcess(args, 0 if found else 1, "", "")
    monkeypatch.setattr(k8s, "run", run)
    cli._retire_inference_service(Model(name="m", namespace="ns", kind="LLMInferenceService"))
    deleted = [c for c in calls if c.startswith("delete")]
    want = ["delete InferenceService m", "delete aigatewayroute aigw-m", "delete aiservicebackend m",
            "delete backend m"]
    assert deleted == (want if retired else [])


def test_undeploy_after_an_upgrade_removes_the_old_routes(monkeypatch):
    # Upgrade, then undeploy with no deploy in between: the InferenceService's direct-route objects go too.
    for k, v in {"MM_CONFIG": str(cli.MM_DIR / "models.yaml"), "MM_AIGATEWAY_ENABLED": "true",
                 "MM_AI_GATEWAY_PROVIDER": "envoy", "MM_LITELLM_ENABLED": "false", "MM_PLATFORM": "kubernetes"}.items():
        monkeypatch.setenv(k, v)
    deleted, gone = [], set()

    def run(*args, stdin=None):
        if args[0] == "delete":
            deleted.append(" ".join(args[1:3])); gone.add(args[1:3])
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:2] == ("get", "InferenceService") and ("InferenceService", "qwen3-5-2b-ovms") not in gone:
            return subprocess.CompletedProcess(args, 0, "model-manager", "")
        old = args[1:3] in (("aigatewayroute", "aigw-qwen3-5-2b-ovms"), ("aiservicebackend", "qwen3-5-2b-ovms"))
        return subprocess.CompletedProcess(args, 0 if old and args[1:3] not in gone else 1, "", "")
    monkeypatch.setattr(k8s, "run", run)
    cli.undeploy(cli.parse_args(["undeploy", "qwen3-5-2b-ovms"]), "qwen3-5-2b-ovms")
    for obj in ("aiservicebackend qwen3-5-2b-ovms", "backend qwen3-5-2b-ovms", "InferenceService qwen3-5-2b-ovms",
                "LLMInferenceService qwen3-5-2b-ovms"):
        assert obj in deleted, (obj, deleted)


def test_runtime_without_a_signal_is_rejected(monkeypatch, tmp_path):
    catalog = tmp_path / "models.yaml"
    catalog.write_text((cli.MM_DIR / "models.yaml").read_text().replace(
        "  openvino:\n", "  openvino:\n    autoscaling: {query: \"\"}\n", 1))
    with pytest.raises(SystemExit):
        _render(monkeypatch, "qwen3-5-2b-ovms", "--max-replicas", "3", catalog=catalog)


def test_only_llmisvc_models_retire_an_inference_service(monkeypatch):
    calls = []
    monkeypatch.setattr(k8s, "run", lambda *a, stdin=None: calls.append(a) or subprocess.CompletedProcess(a, 1, "", ""))
    cli._retire_inference_service(Model(name="m", namespace="ns", kind="InferenceService"))
    assert calls == []  # no lookup: nothing moves the other way


BALLOON = "cpu-policy.model-manager.io/balloon"


@pytest.mark.parametrize("old_cpu, old_balloon, new_cpu, credited", [
    ("8", "vllm-balloon", "8", 2),      # same shape: the new replicas take the freed cores
    ("16", "vllm-balloon", "8", 2),     # smaller new replicas fit too
    ("8", "vllm-balloon", "32", 0),     # bigger new replicas: full check
    ("8", "vllm-balloon-tp2", "8", 0),  # another balloon (tp): full check
    ("", "vllm-balloon", "8", 0),       # unreadable old spec: full check
])
def test_upgrade_credit_needs_the_freed_cores_to_fit(monkeypatch, old_cpu, old_balloon, new_cpu, credited):
    m = Model(name="m", namespace="ns", kind="LLMInferenceService")
    monkeypatch.setattr(cli, "_owned_inference_service", lambda m: Model(name="m", namespace="ns", kind="InferenceService"))
    monkeypatch.setattr(cli, "_running_replicas", lambda m, node="": 2)
    spec = {"spec": {"predictor": {"annotations": {BALLOON: old_balloon},
                                   "model": {"resources": {"requests": {"cpu": old_cpu}}}}}}
    monkeypatch.setattr(k8s, "get_json", lambda *a: spec)
    assert cli._upgrade_credit(m, {BALLOON: "vllm-balloon"}, new_cpu) == credited


def test_no_upgrade_credit_without_an_old_inference_service(monkeypatch):
    monkeypatch.setattr(cli, "_owned_inference_service", lambda m: None)
    assert cli._upgrade_credit(Model(name="m", namespace="ns", kind="LLMInferenceService"), {}, "8") == 0


@pytest.mark.parametrize("node, credited", [("", 2), ("a", 1), ("b", 1), ("c", 0)])
def test_upgrade_credit_counts_only_the_pinned_nodes_pods(monkeypatch, node, credited):
    # A deploy pinned to a node gains only the cores the old pods free on that node.
    monkeypatch.setattr(cli, "_owned_inference_service", lambda m: Model(name="m", namespace="ns", kind="InferenceService"))
    pods = {"items": [{"spec": {"nodeName": n}, "status": {"phase": "Running"}} for n in ("a", "b")]}
    spec = {"spec": {"predictor": {"annotations": {BALLOON: "vllm-balloon"},
                                   "model": {"resources": {"requests": {"cpu": "8"}}}}}}
    monkeypatch.setattr(k8s, "get_json", lambda *a: pods if a[0] == "pods" else spec)
    m = Model(name="m", namespace="ns", kind="LLMInferenceService", node=node)
    assert cli._upgrade_credit(m, {BALLOON: "vllm-balloon"}, "8") == credited


def test_failed_retire_aborts_and_keeps_the_routes(monkeypatch):
    # The old model keeps serving: no route is removed and the replacement is not applied.
    old = Model(name="m", namespace="ns", kind="InferenceService")
    monkeypatch.setattr(cli, "_owned_inference_service", lambda m: old)
    monkeypatch.setattr(k8s, "run", lambda *a, stdin=None: subprocess.CompletedProcess(a, 1, "", "forbidden"))
    deregistered = []
    monkeypatch.setattr(routing, "deregister", deregistered.append)
    with pytest.raises(SystemExit):
        cli._retire_inference_service(Model(name="m", namespace="ns", kind="LLMInferenceService"))
    assert deregistered == []


@pytest.mark.parametrize("llmisvc, isvc, want", [("", "16", "16"), ("12", "16", "12")])
def test_auto_cpu_keeps_the_old_inference_service_cpu(monkeypatch, llmisvc, isvc, want):
    # An OVMS upgrade keeps the old CPU instead of sizing against cores the old pods still hold.
    live = {"LLMInferenceService": llmisvc, "InferenceService": isvc}
    monkeypatch.setattr(k8s, "out", lambda *a: live[a[1]])
    m = Model(name="m", namespace="ns", kind="LLMInferenceService", cpu="auto")
    policy.resolve_auto_cpu(policy.BestEffort(), m, 1)
    assert m.cpu == want


def test_live_scaling_restores_the_query_for_a_fixed_count():
    # --replicas 1 leaves scale_query empty; the live trigger's query makes the re-render match.
    spec = {"scaling": {"minReplicas": 1, "maxReplicas": 3,
                        "keda": {"triggers": [{"metadata": {"query": "q{model=\"m\"}", "threshold": "2"}}]}}}
    m = Model(name="m", namespace="ns", kind="LLMInferenceService", replicas="1")
    assert renderer.live_scaling(m, spec).scale_query == 'q{model="m"}'


def test_switching_to_a_fixed_count_is_a_scaling_only_change():
    # The live autoscaled spec, read back into a --replicas 1 model, renders the same scaling block.
    auto = Model(name="m", namespace="ns", kind="LLMInferenceService", replicas="1", min_replicas="1",
                 max_replicas="3", scale_target="2", scale_query='q{model="$model"}')
    live = yaml.safe_load("spec:\n" + renderer._llmisvc_replicas(auto))["spec"]
    fixed = Model(name="m", namespace="ns", kind="LLMInferenceService", replicas="1")
    assert renderer._llmisvc_replicas(renderer.live_scaling(fixed, live)) == renderer._llmisvc_replicas(auto)


# ── deploy failures: checks before the download, fast failure after the apply ──

def _quiet_deploy(monkeypatch, steps, ready=""):
    """deploy() with the cluster steps recorded and the wait answering `ready` ('' = Ready)."""
    monkeypatch.setattr(cli, "_require_autoscaling", lambda m: None)
    monkeypatch.setattr(cli, "_owned_inference_service", lambda m: None)
    monkeypatch.setattr(k8s, "ensure_storage", lambda *a: steps.append("storage"))
    monkeypatch.setattr(cli, "download_weights", lambda *a: steps.append("download"))
    monkeypatch.setattr(cli, "_install_runtime", lambda m: None)
    monkeypatch.setattr(k8s, "apply", lambda manifest, ssa: steps.append("apply"))
    monkeypatch.setattr(routing, "register", lambda m: None)
    monkeypatch.setattr(routing, "print_endpoints", lambda m: None)
    monkeypatch.setattr(cli, "on_ready", lambda m: steps.append("ready"))
    monkeypatch.setattr(cli, "_wait_ready", lambda m, t: ready)
    monkeypatch.setattr(cli, "_remove", lambda m, wait, timeout: steps.append("rollback"))


@pytest.mark.parametrize("scaling_only, forced, running", [
    (True, False, 2),   # scaling-only: the running replicas keep their place
    (False, False, 0),  # any other change: every replica is checked
    (True, True, 0),    # MM_FORCE_REDEPLOY runs no diff, so nothing is known to have changed
])
def test_changed_spec_runs_the_gate_then_applies(deploy_env, monkeypatch, scaling_only, forced, running):
    steps, asked = [], []
    if forced:
        monkeypatch.setenv("MM_FORCE_REDEPLOY", "1")
    monkeypatch.setattr(k8s, "succeeds", lambda *a: True)  # deployed; the diff (rc 1) reports a change
    monkeypatch.setattr(cli, "_only_scaling_changed", lambda m, render: asked.append(m.name) or scaling_only)
    monkeypatch.setattr(cli, "_running_replicas", lambda m: 2)
    monkeypatch.setattr(cli, "preflight_gate", lambda *a, running=0, amx=False: steps.append(f"gate {running}"))
    _quiet_deploy(monkeypatch, steps)
    cli.deploy(cli.parse_args(["deploy", "qwen3-0-6b", "--max-replicas", "3", "--wait"]), "qwen3-0-6b")
    assert steps == [f"gate {running}", "storage", "download", "apply", "ready"]
    assert bool(asked) == (not forced)


def test_upgrade_credit_reaches_the_gate(deploy_env, monkeypatch):
    # Whatever the policy: the old InferenceService's pods go before the new ones start.
    steps = []
    monkeypatch.setattr(k8s, "succeeds", lambda *a: False)
    monkeypatch.setattr(cli, "_upgrade_credit", lambda m, annotations, cpu: 1)
    monkeypatch.setattr(cli, "preflight_gate", lambda *a, running=0, amx=False: steps.append(f"gate {running}"))
    _quiet_deploy(monkeypatch, steps)
    monkeypatch.setattr(cli, "_retire_inference_service", lambda m: None)
    cli.deploy(cli.parse_args(["deploy", "qwen3-5-2b-ovms", "--wait"]), "qwen3-5-2b-ovms")
    assert steps[0] == "gate 1"


@pytest.mark.parametrize("existed, rolled_back", [(False, True), (True, False)])
def test_failed_wait_exits_1_and_rolls_back_only_a_new_model(deploy_env, monkeypatch, capsys, existed, rolled_back):
    steps = []
    monkeypatch.setattr(k8s, "succeeds", lambda *a: existed)
    monkeypatch.setattr(cli, "_only_scaling_changed", lambda m, render: False)
    monkeypatch.setattr(cli, "preflight_gate", lambda *a, **k: None)
    _quiet_deploy(monkeypatch, steps, ready="pod p cannot be scheduled: 0/2 nodes are available")
    with pytest.raises(SystemExit) as exit_:
        cli.deploy(cli.parse_args(["deploy", "qwen3-0-6b", "--wait"]), "qwen3-0-6b")
    assert exit_.value.code == 1
    assert ("rollback" in steps) == rolled_back
    err = capsys.readouterr().err
    assert "cannot be scheduled" in err and ("rolled back" in err) == rolled_back


def test_wait_ready_stops_on_a_stuck_pod(monkeypatch):
    monkeypatch.setenv("MM_STUCK_GRACE", "0")
    monkeypatch.setattr(cli, "workload_ready", lambda m: "False")
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    pod = {"metadata": {"name": "p"}, "status": {"conditions": [
        {"type": "PodScheduled", "status": "False", "reason": "Unschedulable", "message": "0/2 nodes: Insufficient memory"}]}}
    monkeypatch.setattr(k8s, "get_json", lambda *a: {"items": [pod]})
    assert cli._wait_ready(Model(name="m", namespace="n"), 600) == \
        "pod p cannot be scheduled: 0/2 nodes: Insufficient memory"


class _Gate(policy.BestEffort):
    """A policy that records the replica counts it is asked to place."""
    name = label = "test"

    def __init__(self, counts_running=True, errors=()):
        self.asked, self.counts_running_pods, self.errors = [], counts_running, list(errors)

    def preflight(self, m, tp):
        self.asked.append(m.replicas)
        return self.errors


@pytest.mark.parametrize("running, checked", [(0, ["4"]), (3, ["1"]), (4, [])])
def test_gate_checks_every_replica_up_to_max(monkeypatch, running, checked):
    requested = []
    monkeypatch.setattr(policy, "_request_errors", lambda m, cpu, amx: requested.append(m.replicas) or [])
    gate = _Gate()
    m = Model(name="m", namespace="n", cpu="8", replicas="1", min_replicas="1", max_replicas="4")
    policy.preflight_gate(gate, m, 1, "8", running=running)
    assert gate.asked == checked and requested == checked
    assert m.replicas == "1"  # checked on a copy


def test_gate_counts_every_replica_for_an_allocatable_policy(monkeypatch):
    # kubelet-static checks total allocatable, so the running replicas count too; free-capacity checks do not.
    requested = []
    monkeypatch.setattr(policy, "_request_errors", lambda m, cpu, amx: requested.append(m.replicas) or [])
    gate = _Gate(counts_running=False)
    policy.preflight_gate(gate, Model(name="m", namespace="n", cpu="8", replicas="1", min_replicas="1",
                                      max_replicas="3"), 1, "8", running=2)
    assert gate.asked == ["3"] and requested == ["1"]


@pytest.mark.parametrize("mode, aborted", [("", True), ("warn", False)])
def test_gate_aborts_when_the_max_does_not_fit(monkeypatch, capsys, mode, aborted):
    monkeypatch.setenv("MM_PREFLIGHT", mode)
    monkeypatch.setattr(policy, "_request_errors", lambda m, cpu, amx: [])
    m = Model(name="m", namespace="n", cpu="8", replicas="1", min_replicas="1", max_replicas="3")
    if aborted:
        with pytest.raises(SystemExit):
            policy.preflight_gate(_Gate(errors=["no room"]), m, 1, "8")
    else:
        policy.preflight_gate(_Gate(errors=["no room"]), m, 1, "8")
    assert "cannot place 3 replica(s) (autoscaling up to 3)" in capsys.readouterr().err


@pytest.mark.parametrize("node_cpus, tp, short", [
    ([16], 1, True),       # 2 slots for 3 replicas
    ([8, 8], 1, True),     # 1 slot per node
    ([16, 8], 1, False),   # 2 + 1 slots
    ([24], 2, False),      # TP splits the 8 CPUs a pod requests, it does not multiply them
])
def test_kubelet_static_replica_shortfall_is_an_error(monkeypatch, node_cpus, tp, short):
    nodes = {"items": [{"status": {"allocatable": {"cpu": str(c)}}, "spec": {}} for c in node_cpus]}
    monkeypatch.setattr(k8s, "get_json", lambda *args: nodes)
    monkeypatch.setattr(policy, "probe_node", lambda node="": "")
    errors = policy.KubeletStatic().preflight(Model(name="m", namespace="n", cpu="8", replicas="3"), tp)
    assert bool(errors) == short and (not short or "would stay Pending" in errors[0])


@pytest.mark.parametrize("quantity, size", [("16Gi", 16 * 2**30), ("512Mi", 512 * 2**20), ("2G", 2 * 10**9),
                                            ("1e9", 10**9), ("1048576", 2**20), ("1.5Gi", 3 * 2**29), ("x", 0)])
def test_mem_bytes(quantity, size):
    assert policy.mem_bytes(quantity) == size


def _node(name, cpu, memory, labels=None, taint=False):
    return {"metadata": {"name": name, "labels": labels or {}},
            "spec": {"taints": [{"effect": "NoSchedule"}]} if taint else {},
            "status": {"allocatable": {"cpu": cpu, "memory": memory}}}


def _pod(node, cpu, memory):
    return {"spec": {"nodeName": node, "containers": [{"resources": {"requests": {"cpu": cpu, "memory": memory}}}]}}


@pytest.mark.parametrize("replicas, amx, fits", [
    ("2", False, True),    # a: 64 CPU / 32Gi free -> 2 by memory; b: tainted
    ("3", False, False),
    ("1", True, False),    # only b has AMX, and it is tainted
])
def test_request_check_counts_free_cpu_and_memory_per_node(monkeypatch, replicas, amx, fits):
    nodes = {"items": [_node("a", "96", "64Gi"),
                       _node("b", "96", "512Gi", labels={policy.AMX_LABEL: "true"}, taint=True)]}
    pods = {"items": [_pod("a", "32", "32Gi"), _pod("", "8", "8Gi")]}  # an unscheduled pod takes nothing
    monkeypatch.setattr(k8s, "get_json", lambda *a: pods if a[0] == "pods" else nodes)
    errors = policy._request_errors(Model(name="m", namespace="n", memory="16Gi", replicas=replicas), "8", amx)
    assert (errors == []) == fits
    assert fits or "would stay Pending" in errors[0]


def test_request_check_skips_with_a_warning_when_unreadable(monkeypatch, capsys):
    monkeypatch.setattr(k8s, "get_json", lambda *a: None)
    assert policy._request_errors(Model(name="m", namespace="n", memory="16Gi", replicas="1"), "8", False) == []
    assert "skipping the CPU/memory request check" in capsys.readouterr().err


# ── stuck pods ──

def _waiting(reason, restarts=0, message=""):
    return {"metadata": {"name": "p"}, "status": {"containerStatuses": [
        {"name": "main", "restartCount": restarts, "state": {"waiting": {"reason": reason, "message": message}},
         "lastState": {"terminated": {"exitCode": 137, "reason": "OOMKilled"}}}]}}


@pytest.mark.parametrize("pod, reason, final", [
    (_waiting("ImagePullBackOff", message="not found"), "container main: ImagePullBackOff: not found", False),
    (_waiting("CrashLoopBackOff", restarts=3), "container main keeps crashing (3 restarts, last exit 137 OOMKilled)",
     True),
    (_waiting("CrashLoopBackOff", restarts=1), "", False),  # may still come up
    (_waiting("ContainerCreating"), "", False),
    ({"status": {"phase": "Pending"}}, "", False),
])
def test_stuck_blocker(pod, reason, final):
    from core import stuck
    assert stuck.blocker(pod) == (reason, final)


def test_stuck_waits_out_the_grace(monkeypatch):
    from core import stuck
    clock = [0.0]
    monkeypatch.setattr(stuck.time, "monotonic", lambda: clock[0])
    monkeypatch.setenv("MM_STUCK_GRACE", "60")
    watch, pod = stuck.Watch(), _waiting("ErrImagePull")
    assert watch.stuck([pod]) == ""
    clock[0] = 59
    assert watch.stuck([pod]) == ""
    clock[0] = 60
    assert watch.stuck([pod]).startswith("pod p container main: ErrImagePull")
    watch.stuck([])  # cleared: a new occurrence starts its grace again
    assert watch.stuck([pod]) == ""


@pytest.mark.parametrize("job, pods, want", [
    ({"status": {"succeeded": 1}}, [], ""),
    ({"status": {"conditions": [{"type": "Failed", "status": "True"}]}}, [], "Failed"),
    ({"status": {}}, [_waiting("ImagePullBackOff")], "pod p container main: ImagePullBackOff: "),
])
def test_download_wait_stops_on_success_failure_or_a_stuck_pod(monkeypatch, job, pods, want):
    monkeypatch.setenv("MM_STUCK_GRACE", "0")
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    monkeypatch.setattr(k8s, "get_json", lambda *a: job if a[0] == "job" else {"items": pods})
    assert download._wait_job(Model(name="m", namespace="n"), "download-m") == want


def test_download_wait_times_out(monkeypatch):
    clock = iter(range(0, 10**6, 4000))
    monkeypatch.setattr(download.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    monkeypatch.setattr(k8s, "get_json", lambda *a: {"status": {}} if a[0] == "job" else {"items": []})
    assert download._wait_job(Model(name="m", namespace="n"), "download-m") == f"not finished after {download.TIMEOUT}s"


# ── upgrade: the old InferenceService's pods go before the apply ──

def test_retire_waits_for_the_old_pods(monkeypatch):
    old = Model(name="m", namespace="ns", kind="InferenceService")
    monkeypatch.setattr(cli, "_owned_inference_service", lambda m: old)
    monkeypatch.setattr(routing, "deregister", lambda m: None)
    monkeypatch.setattr(k8s, "run", lambda *a, stdin=None: subprocess.CompletedProcess(a, 0, "", ""))
    left = ["pod/a", "pod/a", ""]
    monkeypatch.setattr(k8s, "out", lambda *a: left.pop(0))
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    cli._retire_inference_service(Model(name="m", namespace="ns", kind="LLMInferenceService"))
    assert left == []


def test_retire_aborts_when_the_old_pods_stay(monkeypatch):
    old = Model(name="m", namespace="ns", kind="InferenceService")
    monkeypatch.setattr(cli, "_owned_inference_service", lambda m: old)
    monkeypatch.setattr(routing, "deregister", lambda m: None)
    monkeypatch.setattr(k8s, "run", lambda *a, stdin=None: subprocess.CompletedProcess(a, 0, "", ""))
    monkeypatch.setattr(k8s, "out", lambda *a: "pod/a")
    clock = iter(range(0, 10**6, 100))
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    with pytest.raises(SystemExit):
        cli._retire_inference_service(Model(name="m", namespace="ns", kind="LLMInferenceService"))


# ── NRI: real free CPUs per NUMA node, on the NUMA sets the node's BalloonsPolicy defines ──

CONTIGUOUS = '{"0": "0-3,8-11", "1": "4-7,12-15"}'


def _leaf(name, numa):
    return {"name": name, "preferCloseToDevices": [f"/sys/devices/system/node/node{numa}"]}


def _tp_policy(*extra):
    """Two NUMA nodes on two packages, as the role generates it: tp1 takes one, tp2 takes both."""
    return [_leaf("pkg0node0", 0), _leaf("pkg1node1", 1),
            {"name": "vllm-balloon-tp1", "componentCreation": "balance-balloons",
             "components": [{"balloonType": "pkg0node0"}, {"balloonType": "pkg1node1"}]},
            {"name": "crosspkgcomb0-tp2", "componentCreation": "all",
             "components": [{"balloonType": "pkg0node0"}, {"balloonType": "pkg1node1"}]},
            {"name": "vllm-balloon-tp2", "componentCreation": "balance-balloons",
             "components": [{"balloonType": "crosspkgcomb0-tp2"}]}, *extra]


def _nri_cluster(monkeypatch, balloons, annotation=CONTIGUOUS, types=None, reserved="cpuset:8-15"):
    """w1: the NUMA CPU lists the role published, `balloons` = cpusets NRI reports, `types` = balloonTypes."""
    nrt = {"zones": [{"name": f"vllm-balloon-tp1[{i}]", "type": "balloon", "attributes": [{"name": "cpuset", "value": c}]}
                     for i, c in enumerate(balloons)] + [
        {"name": "reserved[0]", "type": "balloon", "attributes": [{"name": "cpuset", "value": "8-9"}]}]}
    policy = {"spec": {"reservedResources": {"cpu": reserved}, "balloonTypes": _tp_policy() if types is None else types}}

    def kubectl(*args):
        line = " ".join(args)
        if "numa-cpus" in line:
            return annotation
        if "reservedResources" in line:
            return reserved
        if args[:2] == ("get", nri_policy._NRT_CRD):
            return json.dumps(nrt)
        if args[:2] == ("get", nri_policy._BALLOONSPOLICY_CRD) and args[-1] == "json":
            return json.dumps(policy)
        if "nri\\.intel\\.com/numa-nodes" in line:
            return "2"
        if "nri\\.intel\\.com/physical-cpus" in line:
            return "8"
        if "capacity.cpu" in line:
            return "16"
        return ""
    monkeypatch.setattr(nri_policy, "_kubectl", kubectl)
    monkeypatch.setattr(nri_policy, "advanced_mode_active", lambda: True)  # the role's per-tp balloon types


def test_per_numa_free_counts_each_balloon_where_it_is(monkeypatch):
    # Both busy cores are on NUMA 0: the even split would have said 3 and 3.
    _nri_cluster(monkeypatch, ["0", "1"])
    assert nri_policy._per_numa_free("w1", in_use=2) == {0: 2, 1: 4}


def test_per_numa_free_takes_unreported_pods_from_the_freest_node(monkeypatch):
    _nri_cluster(monkeypatch, ["0"])
    assert nri_policy._per_numa_free("w1", in_use=3) == {0: 3, 1: 2}  # 2 CPUs requested but not in a balloon yet


def test_per_numa_free_counts_a_spilled_balloon(monkeypatch):
    # A balloon on reserved siblings: its request is fully accounted, the reserved CPUs were never free.
    _nri_cluster(monkeypatch, ["2-3,10-11"])
    assert nri_policy._per_numa_free("w1", in_use=4) == {0: 2, 1: 4}


def test_per_numa_free_with_interleaved_numbering(monkeypatch):
    # NUMA 0 = even CPUs, NUMA 1 = odd; no layout is assumed, the published lists decide.
    _nri_cluster(monkeypatch, ["0,2"], annotation='{"0": "0,2,4,6,8,10,12,14", "1": "1,3,5,7,9,11,13,15"}')
    assert nri_policy._per_numa_free("w1", in_use=2) == {0: 2, 1: 4}


def test_no_published_layout_means_no_exact_count(monkeypatch, capsys):
    _nri_cluster(monkeypatch, ["0"], annotation="")
    assert nri_policy._per_numa_free("w1", in_use=1) is None
    nri_policy._usable_per_numa("w1", [8, 8], 1)
    assert "nri.intel.com/numa-cpus not published" in capsys.readouterr().err


@pytest.mark.parametrize("types, balloon, want", [
    (_tp_policy(), "vllm-balloon-tp1", [(0,), (1,)]),
    (_tp_policy(), "vllm-balloon-tp2", [(0, 1)]),
    # SNC2 on two sockets: tp2 stays within a package (localpkg), as the role generates it.
    ([_leaf(f"pkg{p}node{n}", n) for p, n in ((0, 0), (0, 1), (1, 2), (1, 3))] + [
        {"name": "localpkgcomb0-tp2", "components": [{"balloonType": "pkg0node0"}, {"balloonType": "pkg0node1"}]},
        {"name": "localpkgcomb1-tp2", "components": [{"balloonType": "pkg1node2"}, {"balloonType": "pkg1node3"}]},
        {"name": "vllm-balloon-tp2", "componentCreation": "balance-balloons",
         "components": [{"balloonType": "localpkgcomb0-tp2"}, {"balloonType": "localpkgcomb1-tp2"}]}],
     "vllm-balloon-tp2", [(0, 1), (2, 3)]),
    ([{"name": "vllm-balloon-tp1", "components": [{"balloonType": "nowhere"}]}], "vllm-balloon-tp1", None),
    ([{"name": "vllm-balloon-tp1"}], "vllm-balloon-tp1", None),  # a leaf close to no NUMA node
    (_tp_policy(), "vllm-balloon-tp4", None),
])
def test_tp_numa_sets_follow_the_balloon_policy(monkeypatch, types, balloon, want):
    _nri_cluster(monkeypatch, [], types=types)
    assert nri_policy._tp_numa_sets("w1", balloon) == want


@pytest.mark.parametrize("free, sets, cpu, want, placed", [
    ({0: 4, 1: 4}, [(0,), (1,)], 3, 3, 2),
    ({0: 4, 1: 0, 2: 4, 3: 0}, [(0, 1), (2, 3)], 4, 1, 0),  # spread over NUMA nodes would have said 2
    ({0: 8, 1: 8, 2: 2, 3: 2}, [(0, 1), (2, 3)], 8, 3, 2),
    ({0: 6, 1: 2}, [(0,), (1,)], 2, 9, 4),
])
def test_place_on_numa_sets(free, sets, cpu, want, placed):
    assert nri_policy._place(dict(free), sets, cpu, want) == placed


def test_capacity_report_uses_the_real_numa_placement(monkeypatch):
    # 3-core replicas: NUMA 0 has 2 free, NUMA 1 has 4, so one fits (the even split said 0 or 2).
    _nri_cluster(monkeypatch, ["0-1"])
    monkeypatch.setattr(nri_policy, "_balloon_pods_cpu_on_node", lambda node: 2)
    fits, slots, detail = nri_policy._node_capacity_report("w1", 3, 1, 2)
    assert (fits, slots) == (False, 1) and "per-NUMA free=[2, 4]" in detail


def test_auto_cpu_takes_the_freest_numa_set(monkeypatch):
    _nri_cluster(monkeypatch, ["0-1"])
    monkeypatch.setattr(nri_policy, "_balloon_pods_cpu_on_node", lambda node: 2)
    monkeypatch.setattr(nri_policy, "_schedulable_nri_nodes", lambda tp: ["w1"])
    monkeypatch.setattr(nri_policy, "model_cpu", lambda cpu: cpu)
    assert nri_policy.auto_cpu(1) == 4 and nri_policy.auto_cpu(2) == 4  # tp2: 2 per NUMA node x 2


@pytest.mark.parametrize("cpu, tp, fits", [(4, 1, True), (5, 1, False), (8, 2, True), (9, 2, False)])
def test_largest_replica_is_the_best_numa_set(monkeypatch, cpu, tp, fits):
    # Each NUMA node has 4 non-reserved CPUs: tp1 holds 4, tp2 (both nodes) holds 8.
    _nri_cluster(monkeypatch, [])
    largest = nri_policy._largest_replica("w1", tp)
    assert (cpu <= largest) == fits


def test_largest_replica_unknown_without_a_layout(monkeypatch):
    _nri_cluster(monkeypatch, [], annotation="")
    assert nri_policy._largest_replica("w1", 1) is None
