"""Memory and KV-cache sizing (core/sizer.py) and its use by deploy. Configs are trimmed from the real repos."""
from __future__ import annotations

import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import cli, config, k8s, policy, renderer, sizer  # noqa: E402
from core.config import Catalog, Model, Sizing  # noqa: E402
from core.sizer import GIB, Layer, Unsupported  # noqa: E402

LLAMA_8B = {"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
            "hidden_size": 4096, "max_position_embeddings": 131072, "torch_dtype": "bfloat16"}
LLAMA_1B = {**LLAMA_8B, "num_hidden_layers": 16, "head_dim": 64, "hidden_size": 2048}
QWEN3_06B = {"num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
             "max_position_embeddings": 40960, "torch_dtype": "bfloat16"}
QWEN25_15B = {"num_hidden_layers": 28, "num_attention_heads": 12, "num_key_value_heads": 2, "hidden_size": 1536,
              "max_position_embeddings": 32768, "torch_dtype": "bfloat16"}
GEMMA4 = {"model_type": "gemma4", "text_config": {
    "num_hidden_layers": 30, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
    "num_global_key_value_heads": 2, "global_head_dim": 512, "attention_k_eq_v": True, "sliding_window": 1024,
    "max_position_embeddings": 262144, "torch_dtype": "bfloat16",
    "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 5}}
QWEN36 = {"model_type": "qwen3_5_moe", "text_config": {
    "num_hidden_layers": 40, "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
    "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
    "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "max_position_embeddings": 262144,
    "dtype": "bfloat16", "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 10}}
WHISPER = {"model_type": "whisper", "is_encoder_decoder": True, "decoder_layers": 12, "decoder_attention_heads": 12,
           "d_model": 768, "max_target_positions": 448, "max_source_positions": 1500, "torch_dtype": "float32"}
LLM_ARGS = ["--dtype", "bfloat16", "--max-num-seqs", "128", "--max-num-batched-tokens", "2048",
            "--enable-prefix-caching", "--block-size", "128"]
MIB = 1024 ** 2


def gib(x: float) -> int:
    return int(x * GIB)


@pytest.mark.parametrize("tp, kv, memory", [(1, 16, 40), (2, 8, 43), (4, 4, 50)])
def test_dense_memory_by_tp(tp, kv, memory):
    # Weights split across workers; KV heads split too, so only the per-worker overhead grows with tp.
    s = sizer.size(LLAMA_8B, gib(14.96), LLM_ARGS, tp)
    assert (s.kv_gib, s.memory_gi, s.max_model_len) == (kv, memory, 8192)


def test_kv_heads_replicate_beyond_their_count():
    # 2 KV heads at tp 4: each worker holds one, so the per-worker KV does not shrink below a head.
    layers = sizer.layout(QWEN25_15B)
    assert sizer.bytes_per_request(layers, 4, 8192, 2, 2, 128, 2048, True) == 28 * 64 * 128 * 512
    assert sizer.size(QWEN25_15B, gib(2.88), LLM_ARGS, 4).kv_gib == sizer.KV_MIN_GIB


def test_sliding_layers_hold_the_window_plus_a_batch():
    layers = sizer.layout(GEMMA4)
    assert [x.kind for x in layers].count("sliding") == 25
    assert layers[5] == Layer("full", 2, 512)  # k_eq_v: global KV heads, global head size
    # full: 5 x 8192 tokens x 4 KiB; sliding: 25 x (1023 + 2048 tokens + a block) x 8 KiB, all in 1 MiB pages
    assert sizer.bytes_per_request(layers, 1, 8192, 2, 2, 128, 2048, True) == 785 * MIB
    s = sizer.size(GEMMA4, gib(48.07), LLM_ARGS, 1)
    assert (s.kv_gib, s.memory_gi) == (13, 74)


def test_linear_attention_state_pads_the_page():
    layers = sizer.layout(QWEN36)
    # A 2 MiB fp32 state + conv state needs 9 attention blocks of 128 tokens: 1152-token, 2.25 MiB pages.
    one_state = sizer.bytes_per_request(layers, 1, 32768, 2, 2, 128, 32768, False)
    assert one_state == (10 * 29 + 30) * 2304 * 1024
    assert sizer.bytes_per_request(layers, 1, 32768, 2, 2, 128, 32768, True) == one_state + 30 * 2304 * 1024
    args = [*LLM_ARGS, "--no-enable-prefix-caching", "--max-model-len", "32768", "--max-num-batched-tokens", "32768"]
    s = sizer.size(QWEN36, gib(66.97), args, 1)
    assert (s.kv_gib, s.memory_gi, s.max_model_len) == (12, 95, 32768)


def test_encoder_decoder_caches_decoder_and_audio_frames():
    layers = sizer.layout(WHISPER)
    assert [x.kind for x in layers] == ["full"] * 12 + ["cross"] * 12
    args = [*LLM_ARGS, "--dtype", "float32", "--no-enable-prefix-caching"]
    # 12 x (4 self + 16 cross) fp32 blocks: cross-attention is reserved for the 2048 batched tokens, not 1500 frames
    assert sizer.bytes_per_request(layers, 1, 448, 4, 4, 128, 2048, False) == 180 * MIB
    assert sizer.bytes_per_request(layers, 1, 448, 4, 4, 128, 448, False) == 144 * MIB
    s = sizer.size(WHISPER, gib(0.90), args, 1)
    assert (s.kv_gib, s.memory_gi, s.max_model_len) == (3, 8, 448)  # context capped at max_target_positions


def test_draft_model_adds_weights_and_kv():
    s = sizer.size(LLAMA_8B, gib(14.96), LLM_ARGS, 1, draft=(LLAMA_1B, gib(2.30)))
    assert (round(s.weights_gib, 2), s.kv_gib, s.memory_gi) == (17.26, 20, 47)


@pytest.mark.parametrize("extra, kv", [(["--kv-cache-dtype", "fp8_e4m3"], 8), (["--max-num-seqs", "4"], 4),
                                       (["--max-model-len=4096"], 8)])
def test_engine_args_change_the_kv_cache(extra, kv):
    assert sizer.size(LLAMA_8B, gib(14.96), LLM_ARGS + extra, 1).kv_gib == kv


def test_explicit_kv_needs_no_config():
    s = sizer.size(None, gib(14.96), LLM_ARGS, 2, kv_gib=40)
    assert s.kv_gib == 40 and round(s.needed_gib, 2) == 100.96  # 14.96 + 2 x (40 + 3)


def test_fixed_memory_takes_the_kv_that_fits():
    # qwen3-0-6b at a fixed 16Gi: 16 x 8K streams want 14 GiB, 11 GiB fits beside 1.4 GiB weights + 3 GiB overhead.
    assert sizer.size(QWEN3_06B, gib(1.40), LLM_ARGS, 1, memory_gib=16).kv_gib == 11
    assert sizer.size(QWEN3_06B, gib(1.40), LLM_ARGS, 1, memory_gib=64).kv_gib == 14
    assert sizer.size(QWEN3_06B, gib(1.40), LLM_ARGS, 1, memory_gib=4).kv_gib == 14  # too small: the gate reports it


@pytest.mark.parametrize("cfg, weights, args, reason", [
    ({**LLAMA_8B, "kv_lora_rank": 512}, 1, [], "MLA"),
    ({**LLAMA_8B, "layer_types": ["mamba"] * 32}, 1, [], "layer type mamba"),
    ({"num_hidden_layers": 2}, 1, [], "lacks"),
    (LLAMA_8B, 0, [], "no safetensors"),
    (None, 1, [], "gated"),
    (LLAMA_8B, 1, ["--pipeline-parallel-size", "2"], "pipeline"),
])
def test_unsizable_models(cfg, weights, args, reason):
    with pytest.raises(Unsupported, match=reason):
        sizer.size(cfg, weights, args, 1)


@pytest.mark.parametrize("cfg, tp, ok", [(LLAMA_8B, 3, False), (QWEN25_15B, 4, True), (LLAMA_8B, 1, True),
                                         (WHISPER, 4, True), (WHISPER, 5, False)])
def test_topology(cfg, tp, ok):
    assert (sizer.topology_error(cfg, tp) == "") == ok


def test_arg_value_last_wins_in_any_spelling():
    assert sizer.arg_value(["--max_num_seqs", "8", "--max-num-seqs=4"], "--max-num-seqs") == "4"
    assert sizer.arg_value(["--dtype"], "--dtype") == "" and sizer.arg_value([], "--dtype") == ""



@pytest.fixture
def hub(monkeypatch):
    """Sizing on, with the HuggingFace answers of a Llama-8B (draft: Llama-1B); kubectl unreachable."""
    for k, v in {"MM_AUTOSIZE": "true", "MM_PLATFORM": "kubernetes", "MM_LLMISVC_API_VERSION": "v1alpha2",
                 "MM_CPU_POLICY": "best-effort"}.items():
        monkeypatch.setenv(k, v)
    configs = {"meta-llama/Llama-3.1-8B-Instruct": LLAMA_8B, "org/draft": LLAMA_1B}
    monkeypatch.setattr(cli, "hf_config", lambda repo, _: configs.get(repo))
    monkeypatch.setattr(cli, "hf_weight_bytes", lambda repo, _: gib(2.30 if repo == "org/draft" else 14.96))
    monkeypatch.setattr(k8s, "run", lambda *a, stdin=None: subprocess.CompletedProcess(a, 1, "", ""))
    return configs


def _sized(*argv, catalog=None):
    o = cli.parse_args(["deploy", *argv])
    cat = Catalog(catalog or cli.MM_DIR / "models.yaml")
    m = config.resolve(cat, o.positional[1] if len(o.positional) > 1 else "", o)
    cli_args, _ = cli._engine_cli_args(o)
    s = cli._apply_sizing(m, cat, o, cli_args, cli._tensor_parallel(m, cli_args))
    doc = yaml.safe_load(renderer.manifest(m, cat, o, {}, m.cpu, cli_args))
    main = doc["spec"]["template"]["containers"][0]
    return m, s, main, {e["name"]: e.get("value") for e in main["env"]}


def test_hfid_deploy_is_sized(hub):
    m, s, main, env = _sized("--hfid", "meta-llama/Llama-3.1-8B-Instruct", "--category", "llm")
    assert main["resources"]["limits"]["memory"] == main["resources"]["requests"]["memory"] == "40Gi"
    assert env["VLLM_CPU_KVCACHE_SPACE"] == "16"
    assert "--max-model-len" in main["args"] and main["args"][main["args"].index("--max-model-len") + 1] == "8192"


def test_cli_max_model_len_wins_and_is_not_repeated(hub):
    m, s, main, env = _sized("--hfid", "meta-llama/Llama-3.1-8B-Instruct", "--category", "llm",
                             "--arg", "--max-model-len=4096")
    assert m.max_model_len == "" and main["args"].count("--max-model-len=4096") == 1
    assert env["VLLM_CPU_KVCACHE_SPACE"] == "8" and s.max_model_len == 4096


def test_explicit_memory_and_kv_are_kept(hub):
    m, s, main, env = _sized("--hfid", "meta-llama/Llama-3.1-8B-Instruct", "--category", "llm", "--memory", "64Gi",
                             "--env", "VLLM_CPU_KVCACHE_SPACE=40")
    assert main["resources"]["limits"]["memory"] == "64Gi" and env["VLLM_CPU_KVCACHE_SPACE"] == "40"
    assert "--max-model-len" not in main["args"] and round(s.needed_gib, 2) == 57.96


@pytest.mark.parametrize("argv, error", [
    (["--env", "VLLM_CPU_KVCACHE_SPACE=40"], "but no memory"),
    (["--memory", "64Gi", "--env", "VLLM_CPU_KVCACHE_SPACE=40G"], "whole number"),
    (["--tp", "3"], "does not divide"),
])
def test_sizing_rejects(hub, capsys, argv, error):
    with pytest.raises(SystemExit):
        _sized("--hfid", "meta-llama/Llama-3.1-8B-Instruct", "--category", "llm", *argv)
    assert error in capsys.readouterr().err


def test_memory_auto_on_the_cli_is_sized(hub):
    m, s, main, env = _sized("--hfid", "meta-llama/Llama-3.1-8B-Instruct", "--category", "llm", "--memory", "auto")
    assert main["resources"]["limits"]["memory"] == "40Gi"


def test_bad_engine_arg_falls_back(hub, capsys):
    m, s, main, env = _sized("--hfid", "meta-llama/Llama-3.1-8B-Instruct", "--category", "llm",
                             "--arg", "--max-num-seqs=abc")
    assert main["resources"]["limits"]["memory"] == "16Gi" and s.kv_gib == 8
    assert "sizing unavailable" in capsys.readouterr().err


def test_unreadable_config_keeps_static_values_checked_against_the_weights(hub, capsys):
    # A gated repo without HF_TOKEN: no config.json, but its weight size is public.
    m, s, main, env = _sized("--hfid", "org/gated", "--category", "llm")
    assert main["resources"]["limits"]["memory"] == "16Gi" and env["VLLM_CPU_KVCACHE_SPACE"] == "8"
    assert round(s.needed_gib, 2) == 25.96  # 14.96 + 8 + 3
    assert "below the 26.0 GiB" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        policy.memory_gate(m, s)


@pytest.mark.parametrize("argv, fits", [(["--memory", "32Gi"], True), (["--tp", "2", "--memory", "32Gi"], False),
                                        (["--memory", "64Gi", "--env", "VLLM_CPU_KVCACHE_SPACE=40"], True)])
def test_static_values_fit_check(hub, capsys, argv, fits):
    m, s, main, env = _sized("--hfid", "org/gated", "--category", "llm", *argv)
    assert ("below the" not in capsys.readouterr().err) == fits
    if fits:
        policy.memory_gate(m, s)
    else:
        with pytest.raises(SystemExit):
            policy.memory_gate(m, s)


def test_unknown_weights_skip_the_check(hub, monkeypatch, capsys):
    monkeypatch.setattr(cli, "hf_weight_bytes", lambda repo, _: 0)
    m, s, main, env = _sized("--hfid", "org/unknown", "--category", "llm")
    assert s is None and main["resources"]["limits"]["memory"] == "16Gi"
    assert "sizing unavailable" in capsys.readouterr().err


@pytest.mark.parametrize("argv, env", [(["bge-base-en"], {}), (["qwen3-0-6b"], {"MM_AUTOSIZE": "false"})])
def test_sizing_off(hub, monkeypatch, argv, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert _sized(*argv)[1] is None


@pytest.mark.parametrize("model_args, argv, tp, memory", [
    (["--tensor-parallel-size", "2"], [], 2, "43Gi"),           # the catalog's TP, which vLLM gets
    (["--tensor-parallel-size", "2"], ["--tp", "4"], 4, "50Gi"),  # --tp wins, as in the rendered args
    (["--tensor_parallel_size=2"], [], 2, "43Gi"),
    ([], [], 1, "40Gi"),
])
def test_tp_from_the_catalog_args(hub, tmp_path, model_args, argv, tp, memory):
    catalog = tmp_path / "models.yaml"
    data = yaml.safe_load((cli.MM_DIR / "models.yaml").read_text())
    data["models"] = [{"name": "tp", "model_id": "meta-llama/Llama-3.1-8B-Instruct", "category": "llm",
                       "args": model_args}]
    catalog.write_text(yaml.safe_dump(data))
    m, s, main, env = _sized("tp", *argv, catalog=catalog)
    assert (s.tp, main["resources"]["limits"]["memory"]) == (tp, memory)


def test_bad_catalog_tp_rejected(hub, tmp_path, capsys):
    catalog = tmp_path / "models.yaml"
    data = yaml.safe_load((cli.MM_DIR / "models.yaml").read_text())
    data["models"] = [{"name": "tp", "model_id": "org/m", "category": "llm", "args": ["--tensor-parallel-size", "x"]}]
    catalog.write_text(yaml.safe_dump(data))
    with pytest.raises(SystemExit):
        _sized("tp", catalog=catalog)
    assert "positive integer" in capsys.readouterr().err


def test_draft_from_the_catalog_is_sized(hub, tmp_path):
    catalog = tmp_path / "models.yaml"
    data = yaml.safe_load((cli.MM_DIR / "models.yaml").read_text())
    data["models"] = [{"name": "spec", "model_id": "meta-llama/Llama-3.1-8B-Instruct", "draft_model_id": "org/draft",
                       "category": "llm"}]
    catalog.write_text(yaml.safe_dump(data))
    m, s, main, env = _sized("spec", catalog=catalog)
    assert main["resources"]["limits"]["memory"] == "47Gi" and env["VLLM_CPU_KVCACHE_SPACE"] == "20"


def test_version_0191_takes_the_category_kv(hub, monkeypatch):
    monkeypatch.setenv("MM_AUTOSIZE", "false")
    assert _sized("qwen3-0-6b", "--server-version", "0.19.1")[3]["VLLM_CPU_KVCACHE_SPACE"] == "8"


@pytest.mark.parametrize("memory, needed, env, aborts", [
    ("64Gi", 101, {}, True),                         # tp 2 with a tp 1 KV: OOM at load
    ("64Gi", 101, {"MM_PREFLIGHT": "warn"}, False),
    ("64Gi", 101, {"MM_SKIP_PREFLIGHT": "1"}, False),
    ("64Gi", 58, {}, False),
])
def test_memory_gate(monkeypatch, memory, needed, env, aborts):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    s = Sizing(weights_gib=15, kv_gib=40, tp=2, max_model_len=8192, needed_gib=needed, memory_gi=0)
    m = Model(name="m", namespace="n", memory=memory)
    if aborts:
        with pytest.raises(SystemExit):
            policy.memory_gate(m, s)
    else:
        policy.memory_gate(m, s)


def test_hf_token_goes_on_stdin(monkeypatch):
    calls = []
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setattr(config.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw.get("input")))
                        or subprocess.CompletedProcess(cmd, 0, '{"a": 1}', ""))
    config._hf_json.cache_clear()
    assert config.hf_config("org/gated-token-test", "") == {"a": 1}
    cmd, stdin = calls[0]
    assert "hf_secret" not in " ".join(cmd) and "hf_secret" in stdin and "--config" in cmd
    config._hf_json.cache_clear()
