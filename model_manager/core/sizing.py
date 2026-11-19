"""Memory and KV cache for a model with memory unset or `auto`, from its Hugging Face config and weights.
Engine-neutral; each runtime sets its knobs in a `sizing:` block (runtimes/<name>/values.yaml)."""
from __future__ import annotations

import math

from .config import Catalog, Model, hf_json, hf_model_info, is_uint, pick, text
from .ui import info, warn

GIB = 1 << 30
BLOCK = 1024            # a capped context is rounded down to whole blocks
HYBRID_FIT = 0.9        # mixed layer types: vLLM pads its per-group pages (~5% on Gemma 4)


def flag(args: list[str], name: str) -> str:
    """Value of the last `name` flag (--k v or --k=v) in args, "" when absent or name is empty."""
    value = ""
    for n, a in enumerate(args if name else []):
        if a.startswith(name + "="):
            value = a.split("=", 1)[1]
        elif a == name and n + 1 < len(args):
            value = args[n + 1]
    return value


def weight_bytes(model_id: str, proxy: str) -> int:
    """Bytes an engine loads: root *.safetensors (else *.bin); Meta's original/ and Mistral's consolidated.* skipped."""
    files = {s.get("rfilename", ""): s.get("size") or 0 for s in pick(hf_model_info(model_id, proxy), "siblings", [])}
    root = {f: b for f, b in files.items() if "/" not in f}
    for ext in (".safetensors", ".bin"):
        picked = [b for f, b in root.items() if f.endswith(ext) and not f.startswith("consolidated.")]
        if picked:
            return sum(picked)
    return 0


def kv_layout(cfg: dict, dtype_bytes: int = 2, tp: int = 1) -> tuple[int, int]:
    """(bytes per context token, fixed bytes) of one TP rank's KV cache. Only full attention grows with the context;
    sliding windows are fixed, linear and shared-KV layers add none, unknown types count as full (the safe side)."""
    c = pick(cfg, "text_config", {}) or cfg
    layers, heads = int(pick(c, "num_hidden_layers", 0) or 0), int(pick(c, "num_attention_heads", 0) or 0)
    if not (layers and heads):
        return 0, 0
    head_dim, kv_heads = pick(c, "head_dim", 0) or pick(c, "hidden_size", 0) // heads, \
        pick(c, "num_key_value_heads", 0) or heads
    if pick(c, "kv_lora_rank", 0):  # MLA (DeepSeek): one compressed latent per token and layer
        local = full = (c["kv_lora_rank"] + pick(c, "qk_rope_head_dim", 0)) * dtype_bytes
    else:
        local = 2 * -(-kv_heads // tp) * head_dim * dtype_bytes
        full = 2 * -(-(pick(c, "num_global_key_value_heads", 0) or kv_heads) // tp) * \
            (pick(c, "global_head_dim", 0) or head_dim) * dtype_bytes
    own = layers - int(pick(c, "num_kv_shared_layers", 0) or 0)
    types = (pick(c, "layer_types", []) or ["full_attention"] * layers)[:own]
    window = int(pick(c, "sliding_window", 0) or 0)
    sliding = sum(t == "sliding_attention" for t in types) if window else 0
    per_token = full * (len(types) - sliding - sum(t == "linear_attention" for t in types))
    return per_token, window * local * sliding


def size(m: Model, catalog: Catalog, args: list[str], env: dict, tp: int) -> list[str]:
    """Set m.memory (and m.kv_cache_gb for an env-set cache); return the engine args the sizing adds."""
    s = pick(m.rt, "sizing", {})
    kv_knob, context_arg = pick(s, "kv", {}), text(pick(s, "context_arg"))
    kv_env, kv_arg, kv_tokens = text(pick(kv_knob, "env")), text(pick(kv_knob, "arg")), \
        text(pick(kv_knob, "unit", "GiB")) == "tokens"
    proxy = catalog.https_proxy
    cfg = hf_json(f"{m.model_id}/resolve/main/config.json", proxy)
    per_token, fixed = kv_layout(cfg, int(text(pick(s, "kv_dtype_bytes", 2))), tp)
    weights = weight_bytes(m.model_id, proxy) + (weight_bytes(m.draft_model_id, proxy) if m.draft_model_id else 0)
    if not (per_token and weights):  # Hub unreachable or gated without HF_TOKEN: the category defaults, as before
        if m.auto_size == "auto":
            warn(f"memory: auto: {m.model_id} could not be sized from Hugging Face; using {m.memory}.")
        return []

    max_len = flag(args, context_arg)
    context = int(max_len) if is_uint(max_len) else int(pick(pick(cfg, "text_config", {}) or cfg,
                                                              "max_position_embeddings", 0) or 0)
    set_kv = text(env.get(kv_env)) if kv_env else flag(args, kv_arg)   # an explicit cache always wins
    if is_uint(set_kv):
        kv = math.ceil(int(set_kv) * per_token / GIB) if kv_tokens else int(set_kv)
    else:
        cap = int(text(pick(pick(m.cat, "defaults", {}), "kv_cache_gb", 8)))
        kv = min(cap, max(1, math.ceil((context * per_token + fixed) / GIB)))
    fits = int((kv * GIB - fixed) / per_token * (HYBRID_FIT if fixed else 1)) // BLOCK * BLOCK
    extra = []
    if kv_env:
        m.kv_cache_gb = str(kv)                                      # fills ${KV_CACHE_GB} in the category env
    elif kv_arg and not is_uint(set_kv):
        extra += [kv_arg, str(kv * GIB // per_token if kv_tokens else kv)]
    if context_arg and not max_len and context > fits:
        extra += [context_arg, str(fits)]
    # Quantized weights are repacked into the CPU kernel's layout at load, both copies briefly in memory.
    headroom = float(pick(s, "quantized_headroom" if pick(cfg, "quantization_config") else "weight_headroom", 1.2))
    process = int(pick(s, "process_gib", 4))
    # The engine profiles its largest multimodal input at startup; a max-size video costs far more than an image.
    limits = {k: flag(args, f"--limit-mm-per-prompt.{k}") for k in ("image", "video", "audio")}
    present = ["image", "video"] * ("vision_config" in cfg) + ["audio"] * ("audio_config" in cfg)
    towers = int(pick(s, "multimodal_gib", 10)) if any(limits[k] != "0" for k in present) else 0
    video = int(pick(s, "video_gib", 24)) if pick(cfg, "video_token_id") and limits["video"] != "0" else 0
    m.memory = f"{math.ceil(weights * headroom / GIB + tp * kv + process + towers + video)}Gi"
    info(f"  Memory: {m.memory} (sized from Hugging Face: {weights / GIB:.1f} GiB weights, {kv} GiB KV cache"
         f"{f' x {tp} ranks' if tp > 1 else ''}, context {max_len or min(context, fits)})")
    return extra
