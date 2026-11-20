"""vLLM on Intel Arc Pro B-Series (runtime vllm-xpu): a render golden and unit tests.

The golden (tests/golden/vllm-xpu-dry-run.txt) is a snapshot of the current output, not a legacy
reference. Regenerate it with `python tests/test_vllm_xpu.py` after an intended change.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import cli, config, renderer  # noqa: E402
from core.config import Catalog  # noqa: E402
from test_parity import GOLDEN, MM, case, run  # noqa: E402

NAME = "vllm-xpu-gpt-oss-20b"
CASE = case("vllm-xpu-dry-run", "deploy", NAME, "--dry-run")


def _manifest(monkeypatch, name=NAME, chat_template="", **opts):
    monkeypatch.setenv("MM_CONFIG", str(cli.MM_DIR / "models.yaml"))
    monkeypatch.setenv("MM_PLATFORM", "kubernetes")
    monkeypatch.setenv("MM_LLMISVC_API_VERSION", "v1alpha2")
    catalog = Catalog.load()
    o = cli.Options(**opts)
    m = config.resolve(catalog, name, o)
    m.chat_template = chat_template or m.chat_template
    args, _ = cli._engine_cli_args(o, int(m.gpus), renderer.engine_args(m))
    return m, yaml.safe_load(renderer.manifest(m, catalog, o, {}, m.cpu, args))


def test_render_golden(tmp_path):
    assert run(MM / "model-manager", CASE.values[0], tmp_path) == (GOLDEN / "vllm-xpu-dry-run.txt").read_text()


def test_is_a_gpu_llm_inference_service_requesting_the_intel_resource(monkeypatch):
    m, doc = _manifest(monkeypatch)
    c = doc["spec"]["template"]["containers"][0]
    assert doc["kind"] == "LLMInferenceService" and m.runtime == "vllm-xpu"
    assert c["image"] == "vllm/vllm-openai-xpu:v0.33.0"
    assert c["command"] == ["python3", "-m", "vllm.entrypoints.openai.api_server"]
    assert c["resources"]["requests"]["gpu.intel.com/xe"] == c["resources"]["limits"]["gpu.intel.com/xe"] == "1"
    assert c["args"][c["args"].index("--port") + 1] == "8000" and c["startupProbe"]["httpGet"]["port"] == 8000


def test_runs_non_root_with_device_groups_and_a_long_startup_budget(monkeypatch):
    _, doc = _manifest(monkeypatch)
    c = doc["spec"]["template"]["containers"][0]
    assert c["securityContext"]["runAsNonRoot"] is True and c["securityContext"]["runAsUser"] == 1000
    assert doc["spec"]["template"]["securityContext"]["supplementalGroups"] == [44, 990]
    p = c["startupProbe"]
    assert (p["failureThreshold"], p["periodSeconds"], p["timeoutSeconds"]) == (180, 20, 10)


def test_no_amx_affinity_no_pinning_and_no_remote_code(monkeypatch):
    monkeypatch.setenv("MM_REQUIRE_AMX", "true")
    for name in (NAME, "vllm-xpu-llama-3-1-8b"):
        _, doc = _manifest(monkeypatch, name)
        assert "AMXBF16" not in yaml.safe_dump(doc["spec"]["template"].get("affinity", {}))
        assert "annotations" not in doc["metadata"] and "annotations" not in doc["spec"]
        assert "--trust-remote-code" not in doc["spec"]["template"]["containers"][0]["args"]


def test_per_model_args_override_category_and_tp_follows_gpus(monkeypatch):
    _, doc = _manifest(monkeypatch, "vllm-xpu-llama-3-1-8b")
    args = doc["spec"]["template"]["containers"][0]["args"]
    assert args.count("--dtype") == 1 and args[args.index("--dtype") + 1] == "bfloat16"
    assert "--tensor-parallel-size" not in args  # 1 GPU: nothing to inject


def test_chat_template_is_passed_and_mounted(monkeypatch):
    _, doc = _manifest(monkeypatch, chat_template="{{ messages }}")
    c = doc["spec"]["template"]["containers"][0]
    assert c["args"][c["args"].index("--chat-template") + 1] == "/etc/chat-template/chat_template.jinja"
    assert {"name": "chat-template", "mountPath": "/etc/chat-template", "readOnly": True} in c["volumeMounts"]


def test_waits_up_to_the_runtimes_default(monkeypatch):
    m, _ = _manifest(monkeypatch)
    assert config.text(config.pick(m.rt, "wait_timeout")) == "3600"


def test_catalog_entries_use_gpus_and_the_xpu_runtime():
    cat = yaml.safe_load((cli.MM_DIR / "models.yaml").read_text())
    entries = [m for m in cat["models"] if m["name"].startswith("vllm-xpu-")]
    assert len(entries) == 2
    for m in entries:
        assert m["runtime"] == "vllm-xpu" and int(m["gpus"]) >= 1 and "gpu" not in m
        assert not any(a.startswith("--tensor-parallel") for a in m.get("args", []))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        (GOLDEN / "vllm-xpu-dry-run.txt").write_text(run(MM / "model-manager", CASE.values[0], Path(d)))
