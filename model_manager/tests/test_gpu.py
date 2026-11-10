"""GPU runtime (vllm-cuda): parity cases and unit tests.

Kept apart from test_parity.py and test_units.py so that additions there don't
collide with these. The parity cases reuse test_parity's harness. Their goldens
(tests/golden/gpu-*.txt) come from the legacy GPU branch (DCAIDSE8-2342, commit
0d38ae1), which served Llama 3.1 70B on 8x NVIDIA B300 and reproduces every
other golden; regenerate them with `python tests/test_gpu.py <legacy model_manager
dir>` from that branch, with bash 5.2's patsub_replacement off.
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import cli, config, renderer  # noqa: E402
from core.config import Catalog, Model  # noqa: E402
from core.renderer import gpu_vendor  # noqa: E402
from test_parity import BE, DOWNLOAD, GOLDEN, MM, NO_LITELLM_URL, NRI, POOL, _ready, case, run  # noqa: E402

GPU = {"MM_ACCEL_RESOURCE": "nvidia.com/gpu"}
# GPU pods are Burstable since the B300 run: no CPU limit, so kubelet's static CPU manager never pins them.
BURSTABLE_GPU = (r'^(          limits:)\n            cpu: "[^"]+"\n(            memory: \S+\n            nvidia\.com/gpu:)',
                 r"\1\n\2")
# The weight-download Job tolerates the GPU taint too, so it can run where only tainted GPU nodes fit.
GPU_DOWNLOAD_TOLERATION = (r"^(      restartPolicy: Never\n)",
                           r"\1      tolerations:\n        - key: nvidia.com/gpu\n          operator: Exists\n"
                           r"          effect: NoSchedule\n")
# The vendor preference follows the resource the manifest requests, so an ungated --dry-run shows it too.
UNGATED_VENDOR_PREFERENCE = (r"^(    tolerations:\n)",
                             r"    affinity:\n      nodeAffinity:\n"
                             r"        preferredDuringSchedulingIgnoredDuringExecution:\n"
                             r"          - weight: 100\n            preference:\n              matchExpressions:\n"
                             r"                - key: gpu-vendor\n                  operator: In\n"
                             r"                  values:\n                    - nvidia\n\1")
LLAMA_GPU_READY = [[r"get LLMInferenceService llama-3-1-70b-instruct-cuda -n llm-inference -o json$", 0,
                    _ready("MainWorkloadReady")]]

GPU_CASES = [
    # gpu-llama-70b is the manifest that served on B300.
    case("gpu-llama-70b", "deploy", "llama-3-1-70b-instruct-cuda", "--dry-run", env=GPU, deviations=[BURSTABLE_GPU]),
    case("gpu-dry-run-ungated", "deploy", "qwen3-0-6b-cuda", "--dry-run",
         deviations=[BURSTABLE_GPU, UNGATED_VENDOR_PREFERENCE]),
    case("gpu-count-node", "deploy", "qwen3-0-6b-cuda", "--gpus", "2", "--node", "w1", "--dry-run", env={**GPU, **NRI},
         deviations=[BURSTABLE_GPU]),
    case("gpu-explicit-tp", "deploy", "llama-3-1-70b-instruct-cuda", "--tp", "4", "--dry-run", env=GPU,
         deviations=[BURSTABLE_GPU]),
    case("gpu-flag-ignored-on-cpu", "deploy", "qwen3-0-6b", "--gpus", "2", "--dry-run", env=BE),
    case("gpu-gate-refuses", "deploy", "qwen3-0-6b-cuda"),
    case("gpu-bad-count", "deploy", "qwen3-0-6b-cuda", "--gpus", "x", env=GPU),
    case("gpu-deploy-wait", "deploy", "llama-3-1-70b-instruct-cuda", "--wait", env=GPU,
         k8s=DOWNLOAD + POOL + LLAMA_GPU_READY, deviations=[NO_LITELLM_URL, BURSTABLE_GPU, GPU_DOWNLOAD_TOLERATION]),
]


@pytest.mark.parametrize("c", GPU_CASES)
def test_gpu_parity(c, tmp_path, request):
    expected = (GOLDEN / f"{request.node.callspec.id}.txt").read_text()
    for pattern, repl in c["deviations"]:
        expected, n = re.subn(pattern, repl, expected, flags=re.M)
        assert n, f"stale deviation: {pattern}"
    assert run(MM / "model-manager", c, tmp_path) == expected


def test_zero_gpus_rejected():
    with pytest.raises(SystemExit):
        cli.parse_args(["deploy", "m", "--gpus", "0"])


def test_gpu_tp_defaults_to_gpu_count():
    assert cli._engine_cli_args(cli.Options(), 8) == (["--tensor-parallel-size", "8"], 8)
    assert cli._engine_cli_args(cli.Options(), 1) == ([], 1)
    assert cli._engine_cli_args(cli.Options(tp="4"), 8) == (["--tensor-parallel-size", "4"], 4)


def test_catalog_tp_is_not_overridden():
    # A tensor-parallel size in the catalog (category, version or model args) stands; the GPU-count
    # default would render after it and win, since dedupe_args keeps the last occurrence.
    assert cli._engine_cli_args(cli.Options(), 8, ("--tensor-parallel-size", "4")) == ([], 4)
    assert cli._engine_cli_args(cli.Options(), 8, ("--max-model-len", "8192", "--tensor-parallel-size=2")) == ([], 2)
    assert cli._engine_cli_args(cli.Options(), 8, ("--max-model-len", "8192")) == (["--tensor-parallel-size", "8"], 8)
    # --tp on the command line still overrides the catalog.
    assert cli._engine_cli_args(cli.Options(tp="8"), 8, ("--tensor-parallel-size", "4")) == (
        ["--tensor-parallel-size", "8"], 8)
    args, _ = cli._engine_cli_args(cli.Options(), 8, ("--tensor-parallel-size", "4"))
    assert renderer.dedupe_args(["--tensor-parallel-size", "4", *args]) == ["--tensor-parallel-size", "4"]
    with pytest.raises(SystemExit):
        cli._engine_cli_args(cli.Options(), 8, ("--tensor-parallel-size", "x"))


@pytest.mark.parametrize("cluster", ["gpu.intel.com/xe", "amd.com/gpu", "habana.ai/gaudi"])
def test_gate_refuses_another_vendor(monkeypatch, cluster):
    # vllm-cuda runs the CUDA image: on an Intel or AMD rack it must not request those devices.
    monkeypatch.setenv("MM_ACCEL_RESOURCE", cluster)
    m = Model(name="m", namespace="n", runtime="vllm-cuda", gpus="1",
              rt={"device": "gpu", "accelerator_resource": "nvidia.com/gpu"})
    with pytest.raises(SystemExit):
        cli._gpu_gate(m, dry_run=False)
    cli._gpu_gate(m, dry_run=True)  # a preview warns and renders


def test_gate_accepts_the_runtimes_vendor(monkeypatch):
    monkeypatch.setenv("MM_ACCEL_RESOURCE", "nvidia.com/gpu")
    cli._gpu_gate(Model(name="m", namespace="n", runtime="vllm-cuda", gpus="1",
                        rt={"device": "gpu", "accelerator_resource": "nvidia.com/gpu"}), dry_run=False)
    # Another vendor's GPU runtime is gated on its own resource.
    monkeypatch.setenv("MM_ACCEL_RESOURCE", "gpu.intel.com/xe")
    cli._gpu_gate(Model(name="m", namespace="n", runtime="vllm-xpu", gpus="1",
                        rt={"device": "gpu", "accelerator_resource": "gpu.intel.com/xe"}), dry_run=False)


@pytest.mark.parametrize("resource, vendor", [("nvidia.com/gpu", "nvidia"), ("gpu.intel.com/xe", "intel"),
                                              ("habana.ai/gaudi", "intel"), ("amd.com/gpu", "amd"), ("", ""),
                                              ("example.com/fpga", ""), ("nvidia.com", "")])
def test_gpu_vendor_label(resource, vendor):
    assert gpu_vendor(resource) == vendor


def test_accel_resource_prefers_cluster(monkeypatch):
    m = Model(name="m", namespace="n", rt={"device": "gpu", "accelerator_resource": "nvidia.com/gpu"})
    monkeypatch.delenv("MM_ACCEL_RESOURCE", raising=False)
    assert m.gpu and m.accel_resource == "nvidia.com/gpu"  # --dry-run preview default
    monkeypatch.setenv("MM_ACCEL_RESOURCE", "nvidia.com/mig-1g.10gb")
    assert m.accel_resource == "nvidia.com/mig-1g.10gb"
    assert not Model(name="m", namespace="n", rt={}).gpu


@pytest.mark.parametrize("gpus", ["0", "two", ""])
def test_invalid_catalog_gpu_count_refused(monkeypatch, gpus):
    # --gpus is checked by the parser; a catalog or category value only reaches the gate.
    monkeypatch.setenv("MM_ACCEL_RESOURCE", "nvidia.com/gpu")
    m = Model(name="m", namespace="n", runtime="vllm-cuda", rt={"device": "gpu"}, gpus=gpus)
    with pytest.raises(SystemExit):
        cli._gpu_gate(m, dry_run=True)


def test_gpu_gate_is_inert_on_cpu(monkeypatch):
    monkeypatch.delenv("MM_ACCEL_RESOURCE", raising=False)
    cli._gpu_gate(Model(name="m", namespace="n", runtime="vllm", rt={"device": "cpu"}), dry_run=False)


def test_gpu_resolution_precedence(monkeypatch):
    monkeypatch.setenv("MM_CONFIG", str(cli.MM_DIR / "models.yaml"))
    catalog = Catalog.load()
    resolve = lambda name, **o: config.resolve(catalog, name, cli.Options(**o))  # noqa: E731
    assert resolve("llama-3-1-70b-instruct-cuda").gpus == "8"            # models.yaml gpus
    assert resolve("llama-3-1-70b-instruct-cuda", gpus="2").gpus == "2"  # --gpus wins
    assert resolve("qwen3-0-6b", gpus="2").gpus == ""                    # CPU runtime: never set
    assert resolve("", id="org/m", server="vllm-cuda", category="llm").gpus == "1"  # category default


if __name__ == "__main__":
    legacy = Path(sys.argv[1]).resolve() / "model-manager"
    with tempfile.TemporaryDirectory() as d:
        for p in GPU_CASES:
            (GOLDEN / f"{p.id}.txt").write_text(run(legacy, p.values[0], Path(d)))
            print(f"wrote {p.id}")
