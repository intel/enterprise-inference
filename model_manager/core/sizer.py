"""Memory and KV-cache sizing for vLLM on CPU, from a model's HuggingFace config.json.

The KV pool rules follow vLLM 0.28's CPU backend: one VLLM_CPU_KVCACHE_SPACE pool per
TP worker, page sizes unified across layer types, and linear-attention state padded to
the attention page. See docs/reference/memory_sizing.md.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from .config import Sizing

GIB = 1024 ** 3
MAX_MODEL_LEN = 8192
CONCURRENCY = 16
OVERHEAD_GIB = 3
HEADROOM = 1.15
KV_MIN_GIB = 2
# Load-time allowances (measured on Xeon, intel/enterprise-inference#86): a quantized checkpoint is repacked
# into the CPU kernel's layout at load, with both copies briefly resident; vLLM profiles a multimodal model's
# encoders with their largest input at start-up, a full-size video costing the most.
QUANTIZED_REPACK = 0.4    # x the quantized weight bytes
MULTIMODAL_GIB = 10       # vision / audio towers
VIDEO_GIB = 24            # video-capable models (Qwen2.5-VL-3B: ~21 GiB); a minimum
DTYPE_BYTES = {"float32": 4, "float": 4, "bfloat16": 2, "float16": 2, "half": 2}


class Unsupported(Exception):
    """The model or its engine args cannot be sized; the caller keeps the static values."""


@dataclass(frozen=True)
class Layer:
    kind: str                # full | sliding | cross | linear | mla
    kv_heads: int = 0
    head_dim: int = 0
    tokens: int = 0          # sliding: the window; cross: the encoder length
    state: tuple = ()        # linear: (key heads, value heads, key dim, value dim, conv kernel)


def arg_value(args: list[str], flag: str) -> str:
    """The last value of an engine flag (--k v or --k=v; '-' and '_' alike), or ''."""
    want, value = flag.replace("_", "-"), ""
    for i, a in enumerate(args):
        key, _, inline = a.partition("=")
        if key.replace("_", "-") == want:
            value = inline if inline else (args[i + 1] if i + 1 < len(args) else "")
    return value


def human_int(value: str) -> int:
    """An integer as vLLM 0.28 parses --max-model-len and --max-num-batched-tokens: 32K = 32,768 (binary
    suffixes K/M/G/T), 25.6k = 25,600 (decimal k/m/g/t) (vllm/utils/argparse_utils.py human_readable_int)."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([kmgtKMGT])", value.strip())
    if not m:
        return int(value)
    number, suffix = m.groups()
    if suffix.islower():
        return int(float(number) * 1000 ** ("kmgt".index(suffix) + 1))
    return int(number) * 1024 ** ("KMGT".index(suffix) + 1)


def _enabled(args: list[str], flag: str, default: bool) -> bool:
    """A --flag / --no-flag pair, last one wins."""
    on, off = flag, "--no-" + flag[2:]
    hits = [a.replace("_", "-") for a in args if a.replace("_", "-") in (on, off)]
    return hits[-1] == on if hits else default


def text_config(cfg: dict) -> dict:
    return cfg.get("text_config") or cfg


def layout(cfg: dict) -> list[Layer]:
    """The cache-holding layers: decoder self-attention + cross-attention for encoder-decoders,
    else one entry per layer_types item (every layer full attention when it is absent)."""
    t = text_config(cfg)
    try:
        if t.get("is_encoder_decoder"):
            heads = t["decoder_attention_heads"]
            dim = t["d_model"] // heads
            return ([Layer("full", heads, dim)] * t["decoder_layers"]
                    + [Layer("cross", heads, dim, t["max_source_positions"])] * t["decoder_layers"])
        if any(k in t for k in ("state_size", "ssm_state_size")):
            raise Unsupported("state-space attention")
        # Layers that reuse an earlier layer's KV (Gemma 3n / Gemma 4 E2B, E4B: the last num_kv_shared_layers)
        # hold none of their own.
        own = (t.get("num_hidden_layers") or len(t["layer_types"])) - (t.get("num_kv_shared_layers") or 0)
        if t.get("kv_lora_rank"):  # MLA (DeepSeek V2/V3, Kimi): one latent + rope key per token, on every worker
            return [Layer("mla", 1, t["kv_lora_rank"] + (t.get("qk_rope_head_dim") or 0))] * own
        heads = t["num_attention_heads"]
        kv = t.get("num_key_value_heads") or heads
        dim = t.get("head_dim") or t["hidden_size"] // heads
        # Gemma 4: full layers take the global head size, and with k_eq_v the global KV head count.
        full_kv = (t.get("num_global_key_value_heads") or kv) if t.get("attention_k_eq_v") else kv
        kinds = {
            "full_attention": lambda: Layer("full", full_kv, t.get("global_head_dim") or dim),
            "sliding_attention": lambda: Layer("sliding", kv, dim, t["sliding_window"]),
            "linear_attention": lambda: Layer("linear", state=(
                t["linear_num_key_heads"], t["linear_num_value_heads"], t["linear_key_head_dim"],
                t["linear_value_head_dim"], t["linear_conv_kernel_dim"])),
        }
        types = (t.get("layer_types") or ["full_attention"] * t["num_hidden_layers"])[:own]
        unknown = sorted(set(types) - set(kinds))
        if unknown:
            raise Unsupported(f"layer type {', '.join(unknown)}")
        built = {k: kinds[k]() for k in set(types)}
        return [built[k] for k in types]
    except (KeyError, TypeError, ZeroDivisionError) as e:
        raise Unsupported(f"config.json lacks {e}") from None


def context_limit(cfg: dict) -> int:
    t = text_config(cfg)
    for key in ("max_position_embeddings", "max_target_positions", "n_positions", "seq_length", "max_sequence_length"):
        if t.get(key):
            return int(t[key])
    raise Unsupported("context length unknown (no max_position_embeddings in config.json)")


def topology_error(cfg: dict, tp: int) -> str:
    """vLLM's own start-up check: heads split evenly, KV heads split or replicated evenly."""
    t = text_config(cfg)
    heads = t.get("decoder_attention_heads") or t.get("num_attention_heads")
    kv = t.get("num_key_value_heads") or heads
    if not heads or tp == 1:
        return ""
    if heads % tp or (kv % tp and tp % kv):
        return f"--tensor-parallel-size {tp} does not divide {heads} attention heads / {kv} KV heads"
    return ""


def _per_token(layer: Layer, tp: int, kv_bytes: int) -> int:
    """K and V of one token in one layer on one worker; KV heads are replicated when tp exceeds them. MLA keeps
    one latent per token, replicated on every worker (vLLM MLAAttentionSpec)."""
    if layer.kind == "mla":
        return layer.head_dim * kv_bytes
    return 2 * math.ceil(layer.kv_heads / tp) * layer.head_dim * kv_bytes


def _linear_state(layer: Layer, tp: int, conv_bytes: int) -> int:
    """Gated DeltaNet state per request on one worker: fp32 recurrent state (forced on AMX CPUs) + conv state."""
    k_heads, v_heads, k_dim, v_dim, kernel = layer.state
    conv_dim = 2 * k_heads * k_dim + v_heads * v_dim
    return math.ceil(v_heads / tp) * k_dim * v_dim * 4 + math.ceil(conv_dim / tp) * (kernel - 1) * conv_bytes


def bytes_per_request(layers: list[Layer], tp: int, max_len: int, kv_bytes: int, model_bytes: int, block: int,
                      batched_tokens: int, prefix_caching: bool) -> int:
    """KV pool bytes one full-length request holds on one worker.

    Every layer's page is the widest attention layer's block (narrower layers fit more tokens per
    block); with linear layers the block grows until a page also holds one linear state. Layers of one
    kind form KV cache groups of equal size, the smallest kind's count (the largest's when under 1.5x
    it), so a kind whose count is not a multiple is padded with layers that cost as much as its own
    (vLLM 0.28 kv_cache_utils._get_kv_cache_groups_uniform_page_size)."""
    attention = [x for x in layers if x.kind != "linear"]
    linear = [x for x in layers if x.kind == "linear"]
    widest = max(_per_token(x, tp, kv_bytes) for x in attention)
    page = widest * block
    if linear:
        page = math.ceil(_linear_state(linear[0], tp, model_bytes) / page) * page
    kinds: dict[tuple, list[int]] = {}
    for x in attention:
        tokens_per_page = page // _per_token(x, tp, kv_bytes)
        # Cross-attention is reserved for the encoder budget, which defaults to the batched tokens.
        held = {"full": max_len, "mla": max_len, "cross": max(x.tokens, batched_tokens),
                "sliding": min(x.tokens - 1 + batched_tokens, max_len)}[x.kind]
        cost = (math.ceil(held / tokens_per_page) + (x.kind == "sliding")) * page
        kinds.setdefault((x.kind, x.tokens if x.kind == "sliding" else 0), []).append(cost)
    if linear:
        kinds[("linear", 0)] = [(2 if prefix_caching else 1) * page] * len(linear)
    counts = [len(c) for c in kinds.values()]
    group = max(counts) if max(counts) < 1.5 * min(counts) else min(counts)
    return sum(math.ceil(len(c) / group) * group * sum(c) // len(c) for c in kinds.values())


def resident_bytes(cfg: dict | None, weight_bytes: int, args: list[str]) -> int:
    """Weight bytes once loaded: an unquantized checkpoint served with a wider --dtype (bf16 as float32) grows."""
    if not cfg or _quantized(cfg):
        return weight_bytes
    stored = DTYPE_BYTES.get(text_config(cfg).get("torch_dtype") or text_config(cfg).get("dtype"), 0)
    served = DTYPE_BYTES.get(arg_value(args, "--dtype"), stored)
    return weight_bytes * served // stored if stored and served > stored else weight_bytes


def _quantized(cfg: dict | None) -> bool:
    return bool(cfg) and ("quantization_config" in cfg or "quantization_config" in text_config(cfg))


def _mm_limit(args: list[str], modality: str) -> int | None:
    """--limit-mm-per-prompt for one modality: the dotted flag (--limit-mm-per-prompt.image 0), else the JSON
    form ('{"image": 0}', a count or {"count": n}); None when not set."""
    value = arg_value(args, f"--limit-mm-per-prompt.{modality}")
    if not value:
        try:
            value = json.loads(arg_value(args, "--limit-mm-per-prompt") or "{}").get(modality, "")
        except (ValueError, AttributeError):
            value = ""
    if isinstance(value, dict):
        value = value.get("count", "")
    return int(value) if str(value).isdigit() else None


def load_allowance_gib(cfg: dict | None, weight_bytes: int, args: list[str]) -> float:
    """Memory beyond weights, KV and the per-worker runtime that a deploy needs while loading: the repack copy of
    quantized weights, and the encoder profiling of a multimodal model unless --limit-mm-per-prompt sets every
    modality it has to 0 (and video, separately)."""
    if not cfg:
        return 0.0
    extra = QUANTIZED_REPACK * weight_bytes / GIB if _quantized(cfg) else 0.0
    present = ["image", "video"] * ("vision_config" in cfg) + ["audio"] * ("audio_config" in cfg)
    if any(_mm_limit(args, k) != 0 for k in present):
        extra += MULTIMODAL_GIB
    if cfg.get("video_token_id") is not None and _mm_limit(args, "video") != 0:
        extra += VIDEO_GIB
    return extra


def size(cfg: dict | None, weight_bytes: int, args: list[str], tp: int,
         draft: tuple[dict | None, int] | None = None) -> Sizing:
    """Memory and VLLM_CPU_KVCACHE_SPACE for one pod; raises Unsupported when the model cannot be sized."""
    if not weight_bytes:
        raise Unsupported("no safetensors size on HuggingFace")
    if draft and not draft[1]:
        raise Unsupported("no safetensors size for the draft model")
    if cfg is None or (draft and draft[0] is None):
        raise Unsupported("config.json not readable (gated repo without HF_TOKEN?)")
    for flag in ("--pipeline-parallel-size", "--data-parallel-size"):
        if int(arg_value(args, flag) or 1) > 1:
            raise Unsupported(f"{flag} > 1")
    weights = resident_bytes(cfg, weight_bytes, args) + (resident_bytes(draft[0], draft[1], args) if draft else 0)
    layers = layout(cfg) + (layout(draft[0]) if draft else [])
    given = arg_value(args, "--max-model-len")  # auto / -1: vLLM fits the context to the KV cache, sized for 8K here
    max_len = human_int(given) if given not in ("", "auto", "-1") else min(MAX_MODEL_LEN, context_limit(cfg))
    dtype = arg_value(args, "--dtype")
    if dtype in ("", "auto"):
        dtype = text_config(cfg).get("torch_dtype") or text_config(cfg).get("dtype")
    model_bytes = DTYPE_BYTES.get(dtype, 2)
    kv_bytes = 1 if arg_value(args, "--kv-cache-dtype").startswith("fp8") else model_bytes
    per_request = bytes_per_request(
        layers, tp, max_len, kv_bytes, model_bytes, int(arg_value(args, "--block-size") or 128),
        human_int(arg_value(args, "--max-num-batched-tokens") or str(max_len)),
        _enabled(args, "--enable-prefix-caching", True))
    streams = min(CONCURRENCY, int(arg_value(args, "--max-num-seqs") or CONCURRENCY))
    kv_gib = max(KV_MIN_GIB, math.ceil(per_request / GIB), math.ceil(per_request * streams / GIB))
    load = load_allowance_gib(cfg, weight_bytes, args) + (load_allowance_gib(draft[0], draft[1], args) if draft else 0)
    needed = weights / GIB + load + tp * (kv_gib + OVERHEAD_GIB)
    return Sizing(weights_gib=weights / GIB, kv_gib=kv_gib, tp=tp, max_model_len=max_len,
                  needed_gib=needed, memory_gi=math.ceil(needed * HEADROOM), load_gib=load)

