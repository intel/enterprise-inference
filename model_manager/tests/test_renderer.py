"""Tests for manifest renderer and argument deduplication."""
from pathlib import Path
from model_manager.core.renderer import ManifestRenderer

def test_dedupe_args_last_wins():
    raw_args = [
        "--max-num-seqs", "128",
        "--dtype", "bfloat16",
        "--max-num-seqs", "256",
        "--enable-prefix-caching",
        "--tensor-parallel-size=1",
        "--tensor-parallel-size=2",
    ]
    deduped = ManifestRenderer.dedupe_args(raw_args)
    assert "--dtype" in deduped
    assert "bfloat16" in deduped
    assert "--enable-prefix-caching" in deduped
    assert "--tensor-parallel-size=2" in deduped
    assert "--tensor-parallel-size=1" not in deduped
    # --max-num-seqs should keep 256
    seqs_idx = deduped.index("--max-num-seqs")
    assert deduped[seqs_idx + 1] == "256"

def test_build_env_block(tmp_path):
    renderer = ManifestRenderer(tmp_path)
    env_map = {"VLLM_CPU_KVCACHE_SPACE": "8", "USER": "vllm"}
    empty_str = ""
    block = renderer.build_env_block(
        env_map, runtime="vllm",
        http_proxy=empty_str, https_proxy=empty_str, no_proxy=empty_str, hf_token=empty_str
    )
    assert "name: VLLM_CPU_KVCACHE_SPACE" in block
    assert 'value: "8"' in block
    assert "name: USER" in block
    assert 'value: "vllm"' in block

def test_build_affinity_block(tmp_path):
    renderer = ManifestRenderer(tmp_path)
    # AMX enabled
    aff = renderer.build_affinity_block(node="worker-1", require_amx=True)
    assert "feature.node.kubernetes.io/cpu-cpuid.AMXBF16" in aff
    assert "worker-1" in aff

    # NRI max-tp
    nst = {"matchExpressions": [{"values": ["2"]}]}
    aff_nri = renderer.build_affinity_block(node="", require_amx=False, nri_node_selector_term=nst)
    assert "nri.intel.com/max-tp" in aff_nri
    assert '"2"' in aff_nri
