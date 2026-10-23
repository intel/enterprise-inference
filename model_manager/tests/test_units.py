"""Unit tests for behaviour the parity suite cannot observe, mostly fixes over the legacy CLI."""
from __future__ import annotations

import os
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


@pytest.mark.parametrize("flag", ["--tp", "--replicas", "--wait-timeout"])
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
