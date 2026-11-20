"""SGLang on Intel Arc Pro B-Series (runtime sglang-xpu): a render golden and unit tests.

The golden (tests/golden/sglang-xpu-dry-run.txt) is a snapshot of the current output, not a
legacy reference. Regenerate it with `python tests/test_sglang_xpu.py` after an intended change.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import cli, config, renderer  # noqa: E402
from core.config import Catalog  # noqa: E402
from test_parity import GOLDEN, MM, case, run  # noqa: E402

XPU = "sglang-xpu-qwen3-30b-a3b-2507"
CASE = case("sglang-xpu-dry-run", "deploy", XPU, "--dry-run")


def _manifest(monkeypatch, name=XPU, **opts):
    monkeypatch.setenv("MM_CONFIG", str(cli.MM_DIR / "models.yaml"))
    monkeypatch.setenv("MM_PLATFORM", "kubernetes")
    monkeypatch.setenv("MM_LLMISVC_API_VERSION", "v1alpha2")
    catalog = Catalog.load()
    o = cli.Options(**opts)
    m = config.resolve(catalog, name, o)
    args, tp = cli._engine_cli_args(o, int(m.gpus), renderer.engine_args(m))
    return m, yaml.safe_load(renderer.manifest(m, catalog, o, {}, m.cpu, args)), args


def test_render_golden(tmp_path):
    assert run(MM / "model-manager", CASE.values[0], tmp_path) == (GOLDEN / "sglang-xpu-dry-run.txt").read_text()


def test_requests_the_intel_resource_and_derives_tp_from_gpus(monkeypatch):
    m, doc, args = _manifest(monkeypatch)
    res = doc["spec"]["template"]["containers"][0]["resources"]
    assert res["requests"]["gpu.intel.com/xe"] == res["limits"]["gpu.intel.com/xe"] == "4"
    assert args[-2:] == ["--tensor-parallel-size", "4"]
    assert "--tp-size" not in doc["spec"]["template"]["containers"][0]["args"]


def test_is_an_llm_inference_service_on_the_serving_port(monkeypatch):
    _, doc, _ = _manifest(monkeypatch)
    c = doc["spec"]["template"]["containers"][0]
    assert doc["kind"] == "LLMInferenceService"
    assert c["args"][c["args"].index("--port") + 1] == "8000"
    assert c["startupProbe"]["httpGet"]["port"] == 8000


def test_runs_non_root_with_the_runtimes_device_groups(monkeypatch):
    _, doc, _ = _manifest(monkeypatch)
    ctx = doc["spec"]["template"]["containers"][0]["securityContext"]
    assert ctx["runAsNonRoot"] is True and ctx["runAsUser"] == 1000
    assert doc["spec"]["template"]["securityContext"]["supplementalGroups"] == [44, 990]


def test_device_groups_come_from_the_runtime(monkeypatch, tmp_path):
    cat = yaml.safe_load((cli.MM_DIR / "models.yaml").read_text())
    cat["runtimes"]["sglang-xpu"]["device_groups"] = [109, 110]
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(cat))
    monkeypatch.setenv("MM_CONFIG", str(path))
    monkeypatch.setenv("MM_PLATFORM", "kubernetes")
    monkeypatch.setenv("MM_LLMISVC_API_VERSION", "v1alpha2")
    catalog = Catalog.load()
    o = cli.Options()
    m = config.resolve(catalog, XPU, o)
    doc = yaml.safe_load(renderer.manifest(m, catalog, o, {}, m.cpu, []))
    assert doc["spec"]["template"]["securityContext"]["supplementalGroups"] == [109, 110]


def test_exposes_prometheus_metrics(monkeypatch):
    # SGLang answers /metrics with 404 unless --enable-metrics is set; the gateway scheduler polls it.
    _, doc, _ = _manifest(monkeypatch)
    assert "--enable-metrics" in doc["spec"]["template"]["containers"][0]["args"]


def test_startup_probe_allows_the_slow_health_check(monkeypatch):
    _, doc, _ = _manifest(monkeypatch)
    probe = doc["spec"]["template"]["containers"][0]["startupProbe"]
    assert (probe["failureThreshold"], probe["periodSeconds"], probe["timeoutSeconds"]) == (180, 20, 10)


def test_no_amx_affinity_and_no_cpu_pinning(monkeypatch):
    monkeypatch.setenv("MM_REQUIRE_AMX", "true")
    m, doc, _ = _manifest(monkeypatch)
    assert "AMXBF16" not in yaml.safe_dump(doc.get("spec", {}).get("template", {}).get("affinity", {}))
    assert "annotations" not in doc["metadata"]
    # An explicit CPU policy cannot turn pinning on for a GPU runtime.
    _, doc, _ = _manifest(monkeypatch, cpu_policy="nri")
    assert "annotations" not in doc["metadata"] and "annotations" not in doc["spec"]


def test_trust_remote_code_is_per_model_not_per_category(monkeypatch):
    for name in (XPU, "sglang-xpu-qwen3-8b", "sglang-xpu-qwen3-32b", "sglang-xpu-deepseek-r1-distill-qwen-14b"):
        _, doc, _ = _manifest(monkeypatch, name)
        assert "--trust-remote-code" not in doc["spec"]["template"]["containers"][0]["args"]


def test_catalog_entries_use_gpus_and_the_xpu_runtime(monkeypatch):
    cat = yaml.safe_load((cli.MM_DIR / "models.yaml").read_text())
    entries = [m for m in cat["models"] if m["name"].startswith("sglang-xpu-")]
    assert {m["name"] for m in entries} == {XPU, "sglang-xpu-qwen3-8b", "sglang-xpu-qwen3-32b",
                                            "sglang-xpu-deepseek-r1-distill-qwen-14b"}
    for m in entries:
        assert m["runtime"] == "sglang-xpu" and int(m["gpus"]) >= 1 and "gpu" not in m
        assert not any(a.startswith("--tp") for a in m.get("args", []))


def test_a_cpu_model_gets_no_gpu_default(monkeypatch):
    monkeypatch.setenv("MM_CONFIG", str(cli.MM_DIR / "models.yaml"))
    catalog = Catalog.load()
    assert config.resolve(catalog, "qwen3-0-6b", cli.Options()).gpus == ""


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        (GOLDEN / "sglang-xpu-dry-run.txt").write_text(run(MM / "model-manager", CASE.values[0], Path(d)))


def test_sglang_xpu_waits_as_long_as_its_startup_probe():
    # 180 x 20s startup probe = one hour; a 900s default --wait-timeout would roll back a slow first start.
    rt = yaml.safe_load((Path(__file__).resolve().parents[1] / "runtimes/sglang-xpu/values.yaml").read_text())
    assert rt["wait_timeout"] == 3600
    assert rt["container"]["startup_failures"] * rt["container"]["startup_period"] >= rt["wait_timeout"]
