"""Memory and KV-cache sizing for vLLM on CPU, from a model's HuggingFace config.json.

The KV pool rules follow vLLM 0.28's CPU backend: one VLLM_CPU_KVCACHE_SPACE pool per
TP worker, page sizes unified across layer types, and linear-attention state padded to
the attention page. See docs/reference/memory_sizing.md.
"""
from __future__ import annotations

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
DTYPE_BYTES = {"float32": 4, "float": 4, "bfloat16": 2, "float16": 2, "half": 2}


class Unsupported(Exception):
    """The model or its engine args cannot be sized; the caller keeps the static values."""


@dataclass(frozen=True)
class Layer:
    kind: str                # full | sliding | cross | linear
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
        if any(k in t for k in ("kv_lora_rank", "state_size", "ssm_state_size")):
            raise Unsupported("MLA or state-space attention")
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
        types = t.get("layer_types") or ["full_attention"] * t["num_hidden_layers"]
        unknown = sorted(set(types) - set(kinds))
        if unknown:
            raise Unsupported(f"layer type {', '.join(unknown)}")
        built = {k: kinds[k]() for k in set(types)}
        return [built[k] for k in types]
    except (KeyError, TypeError, ZeroDivisionError) as e:
        raise Unsupported(f"config.json lacks {e}") from None


def context_limit(cfg: dict) -> int:
    t = text_config(cfg)
    return int(t.get("max_position_embeddings") or t.get("max_target_positions") or MAX_MODEL_LEN)


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
    """K and V of one token in one layer on one worker; KV heads are replicated when tp exceeds them."""
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
    block); with linear layers the block grows until a page also holds one linear state."""
    attention = [x for x in layers if x.kind != "linear"]
    linear = [x for x in layers if x.kind == "linear"]
    widest = max(_per_token(x, tp, kv_bytes) for x in attention)
    page = widest * block
    if linear:
        page = math.ceil(_linear_state(linear[0], tp, model_bytes) / page) * page
    total = 0
    for x in attention:
        tokens_per_page = page // _per_token(x, tp, kv_bytes)
        # Cross-attention is reserved for the encoder budget, which defaults to the batched tokens.
        held = {"full": max_len, "cross": max(x.tokens, batched_tokens),
                "sliding": min(x.tokens - 1 + batched_tokens, max_len)}[x.kind]
        total += (math.ceil(held / tokens_per_page) + (x.kind == "sliding")) * page
    return total + len(linear) * (2 if prefix_caching else 1) * page


def size(cfg: dict | None, weight_bytes: int, args: list[str], tp: int, draft: tuple[dict, int] | None = None,
         kv_gib: int | None = None, memory_gib: float | None = None) -> Sizing:
    """Memory for one pod. kv_gib, when given, is the explicit VLLM_CPU_KVCACHE_SPACE and needs no config.
    With an explicit memory_gib the KV cache takes what fits beside the weights, up to what the streams need."""
    if not weight_bytes:
        raise Unsupported("no safetensors size on HuggingFace")
    for flag in ("--pipeline-parallel-size", "--data-parallel-size"):
        if int(arg_value(args, flag) or 1) > 1:
            raise Unsupported(f"{flag} > 1")
    weights = weight_bytes + (draft[1] if draft else 0)
    max_len = MAX_MODEL_LEN
    if kv_gib is None:
        if cfg is None or (draft and draft[0] is None):
            raise Unsupported("config.json not readable (gated repo without HF_TOKEN?)")
        layers = layout(cfg) + (layout(draft[0]) if draft else [])
        max_len = int(arg_value(args, "--max-model-len") or min(MAX_MODEL_LEN, context_limit(cfg)))
        dtype = arg_value(args, "--dtype")
        if dtype in ("", "auto"):
            dtype = text_config(cfg).get("torch_dtype") or text_config(cfg).get("dtype")
        model_bytes = DTYPE_BYTES.get(dtype, 2)
        kv_bytes = 1 if arg_value(args, "--kv-cache-dtype").startswith("fp8") else model_bytes
        per_request = bytes_per_request(
            layers, tp, max_len, kv_bytes, model_bytes, int(arg_value(args, "--block-size") or 128),
            int(arg_value(args, "--max-num-batched-tokens") or max_len),
            _enabled(args, "--enable-prefix-caching", True))
        streams = min(CONCURRENCY, int(arg_value(args, "--max-num-seqs") or CONCURRENCY))
        floor = max(KV_MIN_GIB, math.ceil(per_request / GIB))
        kv_gib = max(floor, math.ceil(per_request * streams / GIB))
        if memory_gib is not None:
            room = math.floor((memory_gib - weights / GIB) / tp - OVERHEAD_GIB)
            kv_gib = min(kv_gib, room) if room >= floor else kv_gib
    needed = weights / GIB + tp * (kv_gib + OVERHEAD_GIB)
    return Sizing(weights_gib=weights / GIB, kv_gib=kv_gib, tp=tp, max_model_len=max_len,
                  needed_gib=needed, memory_gi=math.ceil(needed * HEADROOM))


def to_gib(quantity: str) -> float | None:
    """A Kubernetes memory quantity (16Gi, 16G, 512Mi, bytes) in GiB; None when unparseable."""
    m = re.fullmatch(r"([0-9.]+)\s*([KMGTP]i?)?", quantity.strip())
    if not m:
        return None
    unit = m.group(2) or ""
    power = "KMGTP".index(unit[0]) + 1 if unit else 0
    return float(m.group(1)) * (1024 if unit.endswith("i") else 1000) ** power / GIB
