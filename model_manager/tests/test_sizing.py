"""Memory and KV-cache sizing (core/sizer.py) and its use by deploy. Configs are trimmed from the real repos."""
from __future__ import annotations

import math
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
DEEPSEEK_V2_LITE = {"num_hidden_layers": 27, "num_attention_heads": 16, "num_key_value_heads": 16,
                    "kv_lora_rank": 512, "qk_rope_head_dim": 64, "hidden_size": 2048,
                    "max_position_embeddings": 163840, "torch_dtype": "bfloat16"}
GEMMA4_E4B = {"model_type": "gemma4", "vision_config": {}, "audio_config": {},
              "video_token_id": 258884,  # nosec B105 - a vocabulary token id from config.json, not a secret
              "text_config": {**GEMMA4["text_config"], "num_hidden_layers": 42, "num_kv_shared_layers": 18,
                              "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 7}}
MIB = 1024 ** 2
K8 = ["--max-model-len", "8192"]  # with streams=16: the 16 x 8K layouts measured live against vLLM


def gib(x: float) -> int:
    return int(x * GIB)


@pytest.mark.parametrize("tp, kv, memory", [(1, 16, 40), (2, 8, 43), (4, 4, 50)])
def test_dense_memory_by_tp(tp, kv, memory):
    # One request of Llama's full 131,072 tokens: as many as 16 x 8K. Weights split across workers; KV heads split
    # too, so only the per-worker overhead grows with tp.
    s = sizer.size(LLAMA_8B, gib(14.96), LLM_ARGS, tp)
    assert (s.kv_gib, s.memory_gi, s.max_model_len, s.streams) == (kv, memory, 131072, 1)


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
    s = sizer.size(GEMMA4, gib(48.07), [*LLM_ARGS, *K8], 1, streams=16)
    assert (s.kv_gib, s.memory_gi) == (13, 74)
    s = sizer.size(GEMMA4, gib(48.07), LLM_ARGS, 1)  # default: one request of 262,144 tokens
    assert (s.kv_gib, s.memory_gi, s.max_model_len) == (6, 66, 262144)


def test_linear_attention_state_pads_the_page():
    layers = sizer.layout(QWEN36)
    # A 2 MiB fp32 state + conv state needs 9 attention blocks of 128 tokens: 1152-token, 2.25 MiB pages.
    one_state = sizer.bytes_per_request(layers, 1, 32768, 2, 2, 128, 32768, False)
    assert one_state == (10 * 29 + 30) * 2304 * 1024
    assert sizer.bytes_per_request(layers, 1, 32768, 2, 2, 128, 32768, True) == one_state + 30 * 2304 * 1024
    args = [*LLM_ARGS, "--no-enable-prefix-caching", "--max-model-len", "32768", "--max-num-batched-tokens", "32768"]
    s = sizer.size(QWEN36, gib(66.97), args, 1, streams=16)
    assert (s.kv_gib, s.memory_gi, s.max_model_len) == (12, 95, 32768)


def test_encoder_decoder_caches_decoder_and_audio_frames():
    layers = sizer.layout(WHISPER)
    assert [x.kind for x in layers] == ["full"] * 12 + ["cross"] * 12
    args = [*LLM_ARGS, "--dtype", "float32", "--no-enable-prefix-caching"]
    # 12 x (4 self + 16 cross) fp32 blocks: cross-attention is reserved for the 2048 batched tokens, not 1500 frames
    assert sizer.bytes_per_request(layers, 1, 448, 4, 4, 128, 2048, False) == 180 * MIB
    assert sizer.bytes_per_request(layers, 1, 448, 4, 4, 128, 448, False) == 144 * MIB
    s = sizer.size(WHISPER, gib(0.90), args, 1, streams=16)
    assert (s.kv_gib, s.memory_gi, s.max_model_len) == (3, 8, 448)  # context capped at max_target_positions


def test_kinds_are_padded_to_equal_kv_cache_groups():
    # vLLM groups layers by kind in groups of equal size: 8 full + 9 sliding is allocated as 9 + 9.
    full, sliding = Layer("full", 8, 128), Layer("sliding", 8, 128, 1024)
    padded = sizer.bytes_per_request([full] * 8 + [sliding] * 9, 1, 8192, 2, 2, 128, 2048, True)
    assert padded == sizer.bytes_per_request([full] * 9 + [sliding] * 9, 1, 8192, 2, 2, 128, 2048, True)
    # Within 1.5x of each other no kind is padded past the larger count; a multiple needs no padding (Gemma 4).
    assert sizer.bytes_per_request([full] * 5 + [sliding] * 25, 1, 8192, 2, 2, 128, 2048, True) == \
        (5 * 64 + 25 * 25) * 512 * 1024  # 512 KiB pages: 64 per full layer, 25 per sliding one


@pytest.mark.parametrize("cfg, args, factor", [
    (LLAMA_8B, ["--dtype", "float32"], 2),                                   # bf16 weights widened at load
    (LLAMA_8B, ["--dtype", "bfloat16"], 1),
    ({**LLAMA_8B, "torch_dtype": "float32"}, ["--dtype", "bfloat16"], 1),     # narrowing: the files are the bound
    ({**LLAMA_8B, "quantization_config": {"quant_method": "awq"}}, ["--dtype", "float32"], 1),
])
def test_resident_weights_follow_the_serving_dtype(cfg, args, factor):
    assert sizer.resident_bytes(cfg, gib(15), args) == gib(15) * factor


def test_widened_weights_are_sized():
    s = sizer.size(LLAMA_8B, gib(14.96), [*LLM_ARGS, "--dtype", "float32", "--max-model-len", "128",
                                          "--max-num-seqs", "1"], 1)
    assert round(s.weights_gib, 2) == 29.92


def test_draft_model_adds_weights_and_kv():
    s = sizer.size(LLAMA_8B, gib(14.96), LLM_ARGS, 1, draft=(LLAMA_1B, gib(2.30)))
    assert (round(s.weights_gib, 2), s.kv_gib, s.memory_gi) == (17.26, 20, 47)


@pytest.mark.parametrize("extra, kv", [(["--kv-cache-dtype", "fp8_e4m3"], 8), (["--max-num-seqs", "4"], 16),
                                       (["--max-model-len=65536"], 8), (["--max-model-len=4096"], 2)])
def test_engine_args_change_the_kv_cache(extra, kv):
    assert sizer.size(LLAMA_8B, gib(14.96), LLM_ARGS + extra, 1).kv_gib == kv


@pytest.mark.parametrize("seqs", ["8", "256"])
def test_the_default_is_one_request_of_the_full_context(seqs):
    # --max-num-seqs is vLLM's scheduler limit, not a sizing target: the KV holds one 131,072-token request.
    s = sizer.size(LLAMA_8B, gib(14.96), [*LLM_ARGS, "--max-num-seqs", seqs], 1)
    assert (s.streams, s.kv_gib, s.max_model_len) == (1, 16, 131072)


@pytest.mark.parametrize("streams, seqs, kv", [(20, "256", 40), (20, "8", 40), (64, "256", 128)])
def test_kv_streams_set_the_streams_without_a_cap(streams, seqs, kv):
    # kv_streams wins over min(16, --max-num-seqs), with no upper limit.
    s = sizer.size(LLAMA_8B, gib(14.96), [*LLM_ARGS, "--max-model-len=16384", "--max-num-seqs", seqs], 1,
                   streams=streams)
    assert (s.streams, s.kv_gib) == (streams, kv)
    assert s.memory_gi == math.ceil((14.96 + kv + 3) * sizer.HEADROOM)


def test_kv_by_tp_follows_the_kv_heads():
    # Llama-8B has 8 KV heads: the per-worker KV halves with each TP doubling; tp 3 doesn't split the heads.
    s = sizer.size(LLAMA_8B, gib(14.96), [*LLM_ARGS, *K8], 1, streams=256)
    assert s.kv_by_tp == {1: 256, 2: 128, 4: 64, 8: 32}


LLAMA_3B_DRAFT = {**LLAMA_1B, "max_position_embeddings": 32768}


@pytest.mark.parametrize("cfg, args, kw, context, kv", [
    (LLAMA_8B, LLM_ARGS, {}, 131072, 16),                                     # the model's own limit
    (QWEN3_06B, LLM_ARGS, {}, 40960, 5),
    (LLAMA_8B, [*LLM_ARGS, "--max-model-len", "4096"], {}, 4096, 2),          # a given context; KV floor 2 GiB
    (GEMMA4, LLM_ARGS, {}, 262144, 6),                                        # sliding layers hold a window
    (WHISPER, ["--max-num-batched-tokens", "2048"], {}, 448, 2),
    (LLAMA_8B, LLM_ARGS, {"draft": (LLAMA_3B_DRAFT, gib(2.3))}, 32768, 5),    # a draft's shorter limit caps it
    (LLAMA_8B, [*LLM_ARGS, "--no-enable-chunked-prefill"], {}, 2048, 2),      # whole prompts must fit the batch
])
def test_the_context_is_the_models_own(cfg, args, kw, context, kv):
    s = sizer.size(cfg, gib(14.96), args, 1, **kw)
    assert (s.max_model_len, s.kv_gib, s.streams) == (context, kv, 1)


@pytest.mark.parametrize("value, want", [("32K", 32768), ("32k", 32000), ("25.6k", 25600), ("1M", 2 ** 20), ("4096", 4096)])
def test_vllm_numeric_suffixes(value, want):
    assert sizer.human_int(value) == want


@pytest.mark.parametrize("given, length", [("32K", 32768), ("-1", 131072), ("auto", 131072)])
def test_max_model_len_with_suffixes_and_auto(given, length):
    cfg = {**LLAMA_8B, "max_position_embeddings": 131072}
    s = sizer.size(cfg, gib(14.96), [*LLM_ARGS, "--max-model-len", given, "--max-num-batched-tokens", "8K"], 1)
    assert s.max_model_len == length


def test_auto_max_model_len_is_kept_and_kv_sized_for_the_models_context():
    s = sizer.size(LLAMA_8B, gib(14.96), [*LLM_ARGS, "--max-model-len", "auto"], 1)
    assert (s.kv_gib, s.max_model_len) == (16, 131072)


@pytest.mark.parametrize("key", ["max_position_embeddings", "n_positions", "seq_length", "max_sequence_length"])
def test_context_limit_keys(key):
    cfg = {k: v for k, v in LLAMA_8B.items() if k != "max_position_embeddings"}
    assert sizer.size({**cfg, key: 2048}, gib(14.96), LLM_ARGS, 1).max_model_len == 2048


def test_mla_keeps_one_latent_per_token_on_every_worker():
    # kv_lora_rank + qk_rope_head_dim values per token and layer, no separate K and V, not split by TP.
    layers = sizer.layout(DEEPSEEK_V2_LITE)
    assert {x.kind for x in layers} == {"mla"} and len(layers) == 27
    for tp in (1, 2):
        assert sizer.bytes_per_request(layers, tp, 8192, 2, 2, 128, 2048, True) == 27 * 8192 * 576 * 2


@pytest.mark.parametrize("cfg, own", [(GEMMA4_E4B, 24), ({**LLAMA_8B, "num_kv_shared_layers": 8}, 24)])
def test_shared_kv_layers_hold_no_cache(cfg, own):
    assert len(sizer.layout(cfg)) == own


def test_shared_kv_layers_shrink_the_kv():
    full = sizer.layout({**GEMMA4_E4B, "text_config": {**GEMMA4_E4B["text_config"], "num_kv_shared_layers": 0}})
    shared = sizer.layout(GEMMA4_E4B)
    assert [x.kind for x in shared].count("full") == 4 and [x.kind for x in shared].count("sliding") == 20
    assert sizer.bytes_per_request(shared, 1, 8192, 2, 2, 128, 2048, True) < \
        sizer.bytes_per_request(full, 1, 8192, 2, 2, 128, 2048, True)


AWQ = {**LLAMA_8B, "quantization_config": {"quant_method": "awq", "bits": 4}}
VLM = {**LLAMA_8B, "vision_config": {}, "video_token_id": 151656}  # nosec B105 - a vocabulary token id, not a secret


# Qwen-VL encoders take video natively (temporal patches); Gemma's VLM fixture samples frames.
QWEN_VL = {**LLAMA_8B, "vision_config": {"temporal_patch_size": 2},
           "video_token_id": 151656}  # nosec B105 - a vocabulary token id, not a secret
PLE = {"text_config": {"num_hidden_layers": 42, "hidden_size_per_layer_input": 256, "vocab_size_per_layer_input": 262144}}
K128, K32 = 131072, 32768


@pytest.mark.parametrize("cfg, args, context, load", [
    (LLAMA_8B, [], K128, 0),
    (AWQ, [], K128, 0.4 * 5),                                                          # repack copy
    ({**LLAMA_8B, "vision_config": {}}, [], K128, 4),                                  # image tower
    ({**LLAMA_8B, "audio_config": {}}, [], K128, 4),                                   # audio tower
    (VLM, [], K128, 4),                                                                # frame-sampled video: towers only
    (QWEN_VL, [], K128, 4 + 0.2 * 128),                                                # native video: 0.2 GiB per 1K
    (QWEN_VL, [], K32, 4 + 0.2 * 32),
    (QWEN_VL, ["--limit-mm-per-prompt.video", "0"], K128, 4),                          # images only
    (QWEN_VL, ["--limit-mm-per-prompt.image=0", "--limit-mm-per-prompt.video=0"], K128, 0),  # text only
    (QWEN_VL, ["--limit-mm-per-prompt", '{"image": 0, "video": 0}'], K128, 0),        # JSON form
    (QWEN_VL, ["--limit-mm-per-prompt", '{"image": {"count": 0}, "video": 2}'], K128, 4 + 0.2 * 128),
    ({**GEMMA4_E4B, "text_config": {}}, ["--limit-mm-per-prompt.image", "0",          # audio still on
                                         "--limit-mm-per-prompt.video", "0"], K128, 4),
    (PLE, [], K128, 42 * 262144 * 256 * 4 / GIB),                                      # per-layer embeddings, fp32
])
def test_load_allowances(cfg, args, context, load):
    assert sizer.load_allowance_gib(cfg, gib(5), args, context) == pytest.approx(load)


def test_load_allowances_are_sized():
    s = sizer.size(AWQ, gib(5.3), LLM_ARGS, 1)  # Llama-3 AWQ: KV for one 131,072-token request
    assert s.load_gib == pytest.approx(0.4 * 5.3)
    assert s.memory_gi == math.ceil((5.3 + 0.4 * 5.3 + 16 + 3) * sizer.HEADROOM)
    # The video allowance follows the sized context.
    assert sizer.size(QWEN_VL, gib(14.96), LLM_ARGS, 1).memory_gi == math.ceil(
        (14.96 + 4 + 0.2 * 128 + 16 + 3) * sizer.HEADROOM)
    assert sizer.size(QWEN_VL, gib(14.96), [*LLM_ARGS, *K8], 1).load_gib == pytest.approx(4 + 0.2 * 8)


def test_served_context_for_a_fixed_memory():
    assert sizer.served_context(LLAMA_8B, ["--max-model-len", "32K"]) == 32768
    assert sizer.served_context(LLAMA_8B, []) == 131072 and sizer.served_context(None, []) == 0


def test_vlm_category_is_sized(hub):
    m, s, main, env = _sized(*LLAMA[:2], "--category", "vlm", "--memory", "auto")
    assert s and main["resources"]["limits"]["memory"] == f"{s.memory_gi}Gi"


def test_fixed_memory_fit_counts_the_load_allowances(hub, capsys):
    m = Model(name="m", namespace="ns", memory="30Gi")
    assert cli._static_fit(m, gib(5.3), 0, 16, 1, 0.4 * 5.3).needed_gib == pytest.approx(5.3 + 2.12 + 19)
    assert "below" not in capsys.readouterr().err
    assert cli._static_fit(m, gib(5.3), 0, 16, 1, 10).needed_gib > 30
    assert "load 10.0" in capsys.readouterr().err


@pytest.mark.parametrize("cfg, weights, args, reason", [
    ({**LLAMA_8B, "state_size": 16}, 1, [], "state-space"),
    ({**LLAMA_8B, "layer_types": ["mamba"] * 32}, 1, [], "layer type mamba"),
    ({"num_hidden_layers": 2}, 1, [], "lacks"),
    (LLAMA_8B, 0, [], "no safetensors"),
    (None, 1, [], "gated"),
    (LLAMA_8B, 1, ["--pipeline-parallel-size", "2"], "pipeline"),
    ({k: v for k, v in LLAMA_8B.items() if k != "max_position_embeddings"}, 1, [], "context length unknown"),
])
def test_unsizable_models(cfg, weights, args, reason):
    with pytest.raises(Unsupported, match=reason):
        sizer.size(cfg, weights, args, 1)


def test_unknown_draft_weights_cannot_be_sized():
    with pytest.raises(Unsupported, match="draft"):
        sizer.size(LLAMA_8B, gib(14.96), LLM_ARGS, 1, draft=(LLAMA_1B, 0))


@pytest.mark.parametrize("cfg, tp, ok", [(LLAMA_8B, 3, False), (QWEN25_15B, 4, True), (LLAMA_8B, 1, True),
                                         (WHISPER, 4, True), (WHISPER, 5, False)])
def test_topology(cfg, tp, ok):
    assert (sizer.topology_error(cfg, tp) == "") == ok


def test_arg_value_last_wins_in_any_spelling():
    assert sizer.arg_value(["--max_num_seqs", "8", "--max-num-seqs=4"], "--max-num-seqs") == "4"
    assert sizer.arg_value(["--dtype"], "--dtype") == "" and sizer.arg_value([], "--dtype") == ""



@pytest.fixture
def hub(monkeypatch):
    """HuggingFace answers for a Llama-8B (draft: Llama-1B); org/gated has no readable config; kubectl unreachable."""
    for k, v in {"MM_PLATFORM": "kubernetes", "MM_LLMISVC_API_VERSION": "v1alpha2",
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
    s = cli._apply_sizing(m, cat, o, cli_args, cli._tensor_parallel(m, cli_args, o.env))
    doc = yaml.safe_load(renderer.manifest(m, cat, o, {}, m.cpu, cli_args))
    main = doc["spec"]["template"]["containers"][0]
    return m, s, main, {e["name"]: e.get("value") for e in main["env"]}


LLAMA = ("--hfid", "meta-llama/Llama-3.1-8B-Instruct", "--category", "llm")


def _catalog(tmp_path, **entry):
    catalog = tmp_path / "models.yaml"
    data = yaml.safe_load((cli.MM_DIR / "models.yaml").read_text())
    data["models"] = [{"name": "m", "model_id": "meta-llama/Llama-3.1-8B-Instruct", "category": "llm", **entry}]
    catalog.write_text(yaml.safe_dump(data))
    return catalog


def test_memory_auto_is_sized(hub):
    m, s, main, env = _sized(*LLAMA, "--memory", "auto")
    assert main["resources"]["limits"]["memory"] == main["resources"]["requests"]["memory"] == "40Gi"
    assert env["VLLM_CPU_KVCACHE_SPACE"] == "16"
    # Sized for 16 x 8192 tokens; the same 16 GiB pool holds one request of Llama's full 131,072.
    assert main["args"][main["args"].index("--max-model-len") + 1] == "131072"


def test_catalog_memory_auto_is_sized(hub, tmp_path):
    m, s, main, env = _sized("m", catalog=_catalog(tmp_path, memory="auto"))
    assert main["resources"]["limits"]["memory"] == "40Gi"


def test_without_memory_auto_nothing_is_sized(hub, tmp_path):
    # Opt-in for a catalog model: an omitted memory keeps the category default and KV; only their fit is checked.
    m, s, main, env = _sized("m", catalog=_catalog(tmp_path))
    assert main["resources"]["limits"]["memory"] == "16Gi" and env["VLLM_CPU_KVCACHE_SPACE"] == "8"
    assert "--max-model-len" not in main["args"] and round(s.needed_gib, 2) == 25.96  # 14.96 + 8 + 3


@pytest.mark.parametrize("argv, kv", [(["--arg", "--max-model-len=65536"], "8"), (["--arg", "--max-model-len=auto"], "16")])
def test_a_given_max_model_len_wins_and_is_not_repeated(hub, argv, kv):
    m, s, main, env = _sized(*LLAMA, "--memory", "auto", *argv)
    assert m.max_model_len == "" and main["args"].count(argv[1]) == 1 and env["VLLM_CPU_KVCACHE_SPACE"] == kv


def test_explicit_memory_and_kv_are_kept(hub):
    m, s, main, env = _sized(*LLAMA, "--memory", "64Gi", "--env", "VLLM_CPU_KVCACHE_SPACE=40")
    assert main["resources"]["limits"]["memory"] == "64Gi" and env["VLLM_CPU_KVCACHE_SPACE"] == "40"
    assert "--max-model-len" not in main["args"] and round(s.needed_gib, 2) == 57.96


def test_kv_without_memory_deploys_with_the_category_memory(hub):
    # deploy qwen3-0-6b --env VLLM_CPU_KVCACHE_SPACE=20 works as before sizing.
    m, s, main, env = _sized(*LLAMA, "--env", "VLLM_CPU_KVCACHE_SPACE=20")
    assert main["resources"]["limits"]["memory"] == "16Gi" and env["VLLM_CPU_KVCACHE_SPACE"] == "20"


@pytest.mark.parametrize("argv, error", [
    (["--memory", "auto", "--env", "VLLM_CPU_KVCACHE_SPACE=40"], "sizes both"),
    (["--memory", "64Gi", "--env", "VLLM_CPU_KVCACHE_SPACE=40G"], "whole number"),
    (["--memory", "auto", "--tp", "3"], "does not divide"),
    (["--memory", "auto", "--arg", "--max-model-len=abc"], "cannot size"),
    (["--hfid", "org/gated", "--memory", "auto"], "Set HF_TOKEN"),
])
def test_sizing_rejects(hub, capsys, argv, error):
    with pytest.raises(SystemExit):
        _sized(*LLAMA, *argv)
    assert error in capsys.readouterr().err


def test_static_values_below_the_weights_are_flagged(hub, capsys):
    # A gated repo without HF_TOKEN: no config.json, but its weight size is public.
    m, s, main, env = _sized("--hfid", "org/gated", "--category", "llm")
    assert main["resources"]["limits"]["memory"] == "16Gi" and round(s.needed_gib, 2) == 25.96
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
    m, s, main, env = _sized("--hfid", "org/unknown", "--category", "llm", "--memory", "16Gi")
    assert s is None and main["resources"]["limits"]["memory"] == "16Gi" and not capsys.readouterr().err


def test_hfid_memory_defaults_to_auto(hub):
    # --hfid has no catalog memory, so it is sized as with --memory auto.
    m, s, main, env = _sized(*LLAMA)
    assert m.memory_auto_default and main["resources"]["limits"]["memory"] == "40Gi"
    assert env["VLLM_CPU_KVCACHE_SPACE"] == "16" and main["args"][main["args"].index("--max-model-len") + 1] == "131072"


def test_hfid_set_memory_wins(hub):
    m, s, main, env = _sized(*LLAMA, "--memory", "16Gi")
    assert not m.memory_auto and main["resources"]["limits"]["memory"] == "16Gi" and env["VLLM_CPU_KVCACHE_SPACE"] == "8"


@pytest.mark.parametrize("repo, weights, reason", [("org/gated", gib(14.96), "no config.json"),
                                                   ("org/unknown", 0, "no safetensors size")])
def test_hfid_default_falls_back_when_it_cannot_size(hub, monkeypatch, capsys, repo, weights, reason):
    # Unlike an explicit --memory auto (test_sizing_rejects), the default deploys with the category memory, as before.
    monkeypatch.setattr(cli, "hf_weight_bytes", lambda r, _: weights)
    m, s, main, env = _sized("--hfid", repo, "--category", "llm")
    assert not m.memory_auto and main["resources"]["limits"]["memory"] == "16Gi" and env["VLLM_CPU_KVCACHE_SPACE"] == "8"
    assert "--max-model-len" not in main["args"]
    err = capsys.readouterr().err
    assert f"cannot size {m.name}" in err and "using the llm default 16Gi" in err


@pytest.mark.parametrize("argv, memory", [(["--category", "embed"], "16Gi"),
                                          (["--server", "openvino", "--category", "llm"], "16Gi")])
def test_hfid_default_is_only_for_vllm_llm(hub, capsys, argv, memory):
    # memory: auto sizes vLLM llm models on CPU only; elsewhere --hfid keeps the category memory without an error.
    m, s, main, env = _sized("--hfid", "org/other", *argv)
    assert not m.memory_auto_default and not m.memory_auto and main["resources"]["limits"]["memory"] == memory
    assert "CPU only" not in capsys.readouterr().err


def test_unknown_draft_weights_skip_the_check(hub, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "hf_weight_bytes", lambda repo, _: 0 if repo == "org/draft" else gib(14.96))
    assert _sized("m", catalog=_catalog(tmp_path, draft_model_id="org/draft"))[1] is None


@pytest.mark.parametrize("runtime_config", [{"container": {"command": ["ovms"]}}, {"device": "gpu"}])
def test_only_vllm_on_cpu_is_sized(hub, monkeypatch, capsys, runtime_config):
    monkeypatch.setattr(Catalog, "runtime_config", lambda self, name, required=True: runtime_config)
    o = cli.parse_args(["deploy", *LLAMA, "--memory", "auto"])
    cat = Catalog(cli.MM_DIR / "models.yaml")
    m = config.resolve(cat, "", o)
    with pytest.raises(SystemExit):
        cli._apply_sizing(m, cat, o, [], 1)
    assert "vLLM llm and vlm models on CPU only" in capsys.readouterr().err
    o = cli.parse_args(["deploy", *LLAMA])
    assert cli._apply_sizing(config.resolve(cat, "", o), cat, o, [], 1) is None


def test_memory_auto_outside_llm_is_rejected(hub, capsys):
    with pytest.raises(SystemExit):
        _sized("bge-base-en", "--memory", "auto")
    assert "CPU only" in capsys.readouterr().err


@pytest.mark.parametrize("model_args, argv, tp, memory", [
    (["--tensor-parallel-size", "2"], [], 2, "43Gi"),           # the catalog's TP, which vLLM gets
    (["--tensor-parallel-size", "2"], ["--tp", "4"], 4, "50Gi"),  # --tp wins, as in the rendered args
    (["--tensor_parallel_size=2"], [], 2, "43Gi"),
    ([], [], 1, "40Gi"),
])
def test_tp_from_the_catalog_args(hub, tmp_path, model_args, argv, tp, memory):
    m, s, main, env = _sized("m", *argv, catalog=_catalog(tmp_path, memory="auto", args=model_args))
    assert (s.tp, main["resources"]["limits"]["memory"]) == (tp, memory)


def test_bad_catalog_tp_rejected(hub, tmp_path, capsys):
    with pytest.raises(SystemExit):
        _sized("m", catalog=_catalog(tmp_path, args=["--tensor-parallel-size", "x"]))
    assert "positive integer" in capsys.readouterr().err


def test_draft_from_the_catalog_is_sized(hub, tmp_path):
    m, s, main, env = _sized("m", catalog=_catalog(tmp_path, memory="auto", draft_model_id="org/draft"))
    assert main["resources"]["limits"]["memory"] == "47Gi" and env["VLLM_CPU_KVCACHE_SPACE"] == "20"


@pytest.mark.parametrize("argv, entry, streams, kv", [
    (["--kv-streams", "20"], {}, 20, "40"),                                   # CLI
    ([], {"kv_streams": 20}, 20, "40"),                                       # catalog
    (["--kv-streams", "24"], {"kv_streams": 20}, 24, "48"),                   # CLI wins
    ([], {}, 1, "2"),                                                          # default: one 16K request
])
def test_kv_streams_from_the_catalog_and_cli(hub, tmp_path, argv, entry, streams, kv):
    cat = _catalog(tmp_path, memory="auto", args=["--max-model-len", "16384"], **entry)
    m, s, main, env = _sized("m", *argv, catalog=cat)
    assert (s.streams, env["VLLM_CPU_KVCACHE_SPACE"]) == (streams, kv)


@pytest.mark.parametrize("argv, entry", [(["--kv-streams", "0"], {}), ([], {"kv_streams": "x"})])
def test_bad_kv_streams_rejected(hub, tmp_path, capsys, argv, entry):
    with pytest.raises(SystemExit):
        _sized("m", *argv, catalog=_catalog(tmp_path, memory="auto", **entry))
    assert "positive integer" in capsys.readouterr().err


def test_kv_streams_warnings(hub, tmp_path, capsys):
    _sized("m", "--kv-streams", "20", catalog=_catalog(tmp_path, memory="64Gi"))
    assert "has no effect" in capsys.readouterr().err
    _sized("m", "--kv-streams", "20", catalog=_catalog(tmp_path, memory="auto", args=["--max-num-seqs", "8"]))
    assert "exceeds --max-num-seqs 8" in capsys.readouterr().err


def test_kv_streams_joins_the_sizing_inputs_only_when_set(hub, tmp_path):
    cat = _catalog(tmp_path, memory="auto")
    m = config.resolve(Catalog(cat), "m", cli.parse_args(["deploy", "m"]))
    assert _inputs(cat) == cli._sizing_inputs(m, renderer.effective_args(m, [], []), 1)  # unchanged hash by default
    assert _inputs(cat, "--kv-streams", "20") != _inputs(cat)


def test_numa_gate_suggests_the_tp_that_fits_many_streams(hub, tmp_path, monkeypatch, capsys):
    # 256 streams of 8K: 256 GiB KV per worker at tp 1 overflows a 252 GiB NUMA node; at tp 2 each worker holds 128.
    monkeypatch.setattr(policy, "_schedulable_nodes", lambda m: [_numa_node()])
    cat = _catalog(tmp_path, memory="auto", args=K8)
    m, s, main, env = _sized("m", "--kv-streams", "256", catalog=cat)
    with pytest.raises(SystemExit):
        policy.memory_gate(m, s)
    assert "Deploy with --tp 2" in capsys.readouterr().err
    m, s, main, env = _sized("m", "--kv-streams", "256", "--tp", "2", catalog=cat)
    policy.memory_gate(m, s)


def _live(inputs, memory="44Gi", kv="17", context="8192"):
    annotations = {cli.SIZING_INPUTS: inputs, **({cli.SIZED_CONTEXT: context} if context else {})}
    return {"metadata": {"annotations": annotations},
            "spec": {"template": {"containers": [{"resources": {"limits": {"memory": memory}},
                                                  "env": [{"name": "VLLM_CPU_KVCACHE_SPACE", "value": kv}]}]}}}


def _inputs(catalog, *argv):
    """The sizing inputs a deploy of catalog model m with argv records."""
    o = cli.parse_args(["deploy", "m", *argv])
    cat = Catalog(catalog)
    m = config.resolve(cat, "m", o)
    cli_args, _ = cli._engine_cli_args(o)
    return cli._sizing_inputs(m, renderer.effective_args(m, cli_args, o.env), cli._tensor_parallel(m, cli_args, o.env))


def test_a_sized_deploy_records_its_inputs(hub, tmp_path):
    catalog = _catalog(tmp_path, memory="auto")
    m, s, main, env = _sized("m", catalog=catalog)
    assert m.sizing == {cli.SIZING_INPUTS: _inputs(catalog), cli.SIZED_CONTEXT: "131072"}
    doc = yaml.safe_load(renderer.manifest(m, Catalog(catalog), cli.parse_args(["deploy", "m"]), {}, m.cpu, []))
    assert doc["metadata"]["annotations"] == m.sizing


def test_unsized_manifests_carry_no_sizing_annotation(hub):
    m, s, main, env = _sized(*LLAMA, "--memory", "64Gi")
    assert m.sizing == {}


def test_a_deployed_model_with_unchanged_inputs_keeps_its_sizing(hub, monkeypatch, tmp_path):
    # Re-deploys (e.g. an Ansible re-run with the Hub unreachable) render the live values, not a fallback.
    catalog = _catalog(tmp_path, memory="auto")
    monkeypatch.setattr(k8s, "get_json", lambda *a: _live(_inputs(catalog)))
    monkeypatch.setattr(cli, "hf_config", lambda *a: pytest.fail("no HuggingFace call"))
    m, s, main, env = _sized("m", catalog=catalog)
    assert main["resources"]["limits"]["memory"] == "44Gi" and env["VLLM_CPU_KVCACHE_SPACE"] == "17"
    assert main["args"][main["args"].index("--max-model-len") + 1] == "8192" and s is None


@pytest.mark.parametrize("live_argv, argv", [
    ([], ["--tp", "2"]),                                  # TP changed: sized at tp=1, now two workers
    ([], ["--arg", "--max-model-len=16384"]),             # context changed
    ([], ["--server-version", "0.28.0"]),                 # a different image
])
def test_changed_inputs_re_size_a_deployed_model(hub, monkeypatch, tmp_path, live_argv, argv):
    catalog = _catalog(tmp_path, memory="auto", servers={"vllm": {"versions": ["0.24.0", "0.28.0"],
                                                                  "default": "0.24.0"}})
    monkeypatch.setattr(k8s, "get_json", lambda *a: _live(_inputs(catalog, *live_argv), "40Gi", "16"))
    m, s, main, env = _sized("m", *argv, catalog=catalog)
    assert s is not None and m.sizing[cli.SIZING_INPUTS] == _inputs(catalog, *argv)  # sized again, for memory_gate


def test_a_deployed_model_without_recorded_inputs_is_re_sized(hub, monkeypatch, tmp_path):
    monkeypatch.setattr(k8s, "get_json", lambda *a: {"spec": _live("")["spec"]})
    m, s, main, env = _sized("m", catalog=_catalog(tmp_path, memory="auto"))
    assert main["resources"]["limits"]["memory"] == "40Gi"


def test_memory_auto_on_the_cli_re_sizes_a_deployed_model(hub, monkeypatch, tmp_path):
    catalog = _catalog(tmp_path, memory="auto")
    monkeypatch.setattr(k8s, "get_json", lambda *a: _live(_inputs(catalog, "--memory", "auto")))
    m, s, main, env = _sized("m", "--memory", "auto", catalog=catalog)
    assert main["resources"]["limits"]["memory"] == "40Gi" and env["VLLM_CPU_KVCACHE_SPACE"] == "16"


def test_fixed_memory_deploys_get_the_topology_check(hub, capsys):
    with pytest.raises(SystemExit):
        _sized(*LLAMA, "--memory", "64Gi", "--tp", "3")
    assert "does not divide" in capsys.readouterr().err


def test_openshift_additional_args_are_sized(hub, monkeypatch):
    # OpenShift's template appends VLLM_ADDITIONAL_ARGS after the rendered flags, so those win in vLLM.
    monkeypatch.setenv("MM_PLATFORM", "openshift")
    o = cli.parse_args(["deploy", *LLAMA, "--memory", "auto", "--env",
                        "VLLM_ADDITIONAL_ARGS=--dtype float32 --tensor-parallel-size 2"])
    cat = Catalog(cli.MM_DIR / "models.yaml")
    m = config.resolve(cat, "", o)
    cli_args, _ = cli._engine_cli_args(o)
    tp = cli._tensor_parallel(m, cli_args, o.env)
    s = cli._apply_sizing(m, cat, o, cli_args, tp)
    assert tp == 2 and round(s.weights_gib, 2) == 29.92  # bf16 weights served as float32


def test_version_0191_takes_the_category_kv(hub):
    assert _sized("qwen3-0-6b", "--server-version", "0.19.1")[3]["VLLM_CPU_KVCACHE_SPACE"] == "8"


@pytest.mark.parametrize("memory, needed, env, aborts", [
    ("64Gi", 101, {}, True),                         # tp 2 with a tp 1 KV: OOM at load
    ("64Gi", 101, {"MM_PREFLIGHT": "warn"}, False),
    ("64Gi", 101, {"MM_NRI_PREFLIGHT": "warn"}, False),          # the legacy override, as for the CPU gate
    ("64Gi", 101, {"MM_SKIP_PREFLIGHT": "1"}, False),
    ("64Gi", 58, {}, False),
])
def test_memory_gate(monkeypatch, memory, needed, env, aborts):
    monkeypatch.setattr(policy, "_schedulable_nodes", lambda m: [])
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    s = Sizing(weights_gib=15, kv_gib=40, tp=2, max_model_len=8192, needed_gib=needed, memory_gi=0)
    m = Model(name="m", namespace="n", memory=memory)
    if aborts:
        with pytest.raises(SystemExit):
            policy.memory_gate(m, s)
    else:
        policy.memory_gate(m, s)


def _numa_node(numa_nodes="6", max_tp="4", capacity="1584035764Ki"):
    labels = {k: v for k, v in (("nri.intel.com/numa-nodes", numa_nodes), ("nri.intel.com/max-tp", max_tp)) if v}
    return {"metadata": {"name": "w1", "labels": labels}, "spec": {}, "status": {"capacity": {"memory": capacity}}}


@pytest.mark.parametrize("weights, kv, tp, node, error", [
    (15, 16, 1, _numa_node(), None),                       # Llama-8B: 34 GiB per worker in a 252 GiB NUMA node
    (280, 16, 1, _numa_node(), "--tp 2"),                  # 299 GiB per worker: two workers hold 140 GiB of weights each
    (600, 16, 1, _numa_node(), "--tp 4"),
    (600, 16, 1, _numa_node(max_tp="2"), "No TP the node supports"),
    (280, 240, 1, _numa_node(), "No TP the node supports"),  # the KV alone fills the NUMA node
    (280, 16, 2, _numa_node(), None),                      # already split: 159 GiB per worker
    (280, 16, 1, _numa_node(numa_nodes=""), None),         # no published NUMA layout: not checked
])
def test_numa_gate(monkeypatch, capsys, weights, kv, tp, node, error):
    """The node is this cluster's: 6 NUMA nodes (SNC3 on 2 sockets) of 252 GiB, max-tp 4."""
    monkeypatch.setattr(policy, "_schedulable_nodes", lambda m: [node])
    s = Sizing(weights_gib=weights, kv_gib=kv, tp=tp, max_model_len=8192,
               needed_gib=weights + tp * (kv + sizer.OVERHEAD_GIB), memory_gi=0)
    m = Model(name="m", namespace="n", memory="2000Gi")
    if error is None:
        policy.memory_gate(m, s)
        return
    with pytest.raises(SystemExit):
        policy.memory_gate(m, s)
    err = capsys.readouterr().err
    assert "per TP worker" in err and "a NUMA node has 252 GiB" in err and error in err


@pytest.mark.parametrize("nodes, weights, fits", [
    ([_numa_node(numa_nodes="2", max_tp="2", capacity="1Ti")], 280, True),            # 2 x 512 GiB, no SNC
    ([_numa_node(numa_nodes="1", max_tp="1", capacity="256Gi")], 280, False),         # one socket, one NUMA node
    ([_numa_node(), _numa_node(numa_nodes="2", max_tp="2", capacity="1Ti")], 280, True),  # mixed: the 512 GiB node
])
def test_numa_gate_reads_each_nodes_layout(monkeypatch, nodes, weights, fits):
    monkeypatch.setattr(policy, "_schedulable_nodes", lambda m: nodes)
    s = Sizing(weights_gib=weights, kv_gib=16, tp=1, max_model_len=8192, needed_gib=weights + 19, memory_gi=0)
    m = Model(name="m", namespace="n", memory="2000Gi")
    if fits:
        policy.memory_gate(m, s)
    else:
        with pytest.raises(SystemExit):
            policy.memory_gate(m, s)


@pytest.mark.parametrize("spec", [{"unschedulable": True}, {"taints": [{"effect": "NoExecute"}]},
                                  {"taints": [{"effect": "NoSchedule"}]}])
def test_numa_gate_ignores_nodes_a_pod_cannot_land_on(monkeypatch, spec):
    # A cordoned or tainted node with larger NUMA nodes does not let a worker through that fits nowhere else.
    big = {**_numa_node(numa_nodes="2", max_tp="2", capacity="1Ti"), "spec": spec}
    monkeypatch.setattr(k8s, "get_json", lambda *a: {"items": [_numa_node(), big]})
    s = Sizing(weights_gib=280, kv_gib=16, tp=1, max_model_len=8192, needed_gib=299, memory_gi=0)
    with pytest.raises(SystemExit):
        policy.memory_gate(Model(name="m", namespace="n", memory="400Gi"), s)


def test_numa_gate_is_inconclusive_with_an_unlabeled_node(monkeypatch):
    # An eligible node without the NRI labels (e.g. a 512 GiB single-NUMA host) may still fit the worker.
    unlabeled = _numa_node(numa_nodes="", max_tp="", capacity="512Gi")
    monkeypatch.setattr(policy, "_schedulable_nodes", lambda m: [_numa_node(), unlabeled])
    s = Sizing(weights_gib=280, kv_gib=16, tp=1, max_model_len=8192, needed_gib=299, memory_gi=0)
    policy.memory_gate(Model(name="m", namespace="n", memory="400Gi"), s)


def test_numa_gate_warn_mode_proceeds(monkeypatch, capsys):
    monkeypatch.setenv("MM_PREFLIGHT", "warn")
    monkeypatch.setattr(policy, "_schedulable_nodes", lambda m: [_numa_node()])
    s = Sizing(weights_gib=280, kv_gib=16, tp=1, max_model_len=8192, needed_gib=299, memory_gi=0)
    policy.memory_gate(Model(name="m", namespace="n", memory="400Gi"), s)
    assert "--tp 2" in capsys.readouterr().err


def test_weights_are_the_repo_root_safetensors(monkeypatch):
    files = [("model-00001-of-00002.safetensors", 4), ("model-00002-of-00002.safetensors", 4),
             ("original/consolidated.safetensors", 8), ("onnx/model.safetensors", 8), ("consolidated.safetensors", 8),
             ("pytorch_model.bin", 8)]
    monkeypatch.setattr(config, "_hf_json", lambda url, proxy, token: {"siblings": [{"rfilename": n, "size": gib(g)}
                                                                             for n, g in files]})
    assert config.hf_weight_bytes("org/m", "") == gib(8)


def test_an_unreachable_hub_is_tried_once(monkeypatch):
    calls = []
    monkeypatch.setattr(config.subprocess, "run", lambda cmd, **kw: calls.append(cmd)
                        or subprocess.CompletedProcess(cmd, 28, "", ""))
    monkeypatch.setattr(config, "_HUB_UNREACHABLE", set())
    config._hf_json.cache_clear()
    assert config.hf_config("org/a", "http://proxy:1") is None and config.hf_weight_bytes("org/a", "http://proxy:1") == 0
    assert len(calls) == 1
    config._hf_json.cache_clear()


def test_memory_auto_names_an_unreachable_hub(hub, monkeypatch, capsys):
    monkeypatch.setattr(cli, "hub_unreachable", lambda proxy: True)
    monkeypatch.setattr(cli, "hf_weight_bytes", lambda *a: 0)
    with pytest.raises(SystemExit):
        _sized(*LLAMA, "--memory", "auto")
    assert "Hugging Face is unreachable" in capsys.readouterr().err


def test_hf_token_goes_on_stdin(monkeypatch):
    calls = []
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setattr(config.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw.get("input")))
                        or subprocess.CompletedProcess(cmd, 0, '{"a": 1}', ""))
    config._hf_json.cache_clear()
    assert config.hf_config("org/gated-token-test", "") == {"a": 1}
    cmd, stdin = calls[0]
    assert "hf_secret" not in " ".join(cmd) and stdin == "Authorization: Bearer hf_secret\n" and ["-H", "@-"] == cmd[-3:-1]
    config._hf_json.cache_clear()
