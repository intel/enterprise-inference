# Memory & KV-Cache Sizing

## What it does

`memory: auto` in a model's catalog entry, or `--memory auto` on the command line, makes
`model-manager deploy` work out a vLLM chat model's (category `llm`) pod memory and KV
cache from the model itself, like `cpu: auto` does for CPUs. It reads:

- the model's `config.json` on Hugging Face, for the layer layout;
- the safetensors index or file sizes, for the weight size;
- the engine args the model is served with.

Sizing is opt-in: a model without `memory: auto` deploys with its set `memory` (or the
category default) and KV cache, as before; only their [fit](#pre-flight-checks) is checked.

> [!WARNING]
> For a **gated** repo (Llama, Gemma, Mistral) `config.json` is gated too. Without
> `HF_TOKEN`, `memory: auto` cannot size the model and the deploy stops with
> `memory: auto cannot size …: config.json not readable (gated repo without HF_TOKEN?)`.
> Set `HF_TOKEN`, or set `memory:` for that model.

Nothing is downloaded for this: only those small JSON documents are fetched, before
the weight download starts.

The result sets three things in the manifest:

| Setting | Meaning |
|---|---|
| `resources.requests.memory` = `limits.memory` | The pod's memory |
| `VLLM_CPU_KVCACHE_SPACE` | KV cache per vLLM worker, in whole GiB |
| `--max-model-len` | The longest request (prompt + output) the model accepts |

The deploy prints the calculation:

```
[INFO ]    Memory: 40Gi (auto) | needs 34.0 GiB = weights 15.0 + KV 1 x 16 + overhead 1 x 3, x 1.15 headroom | KV for up to 16 x 8192 tokens
```

## Where the values come from

**Memory**, first match wins:

| # | Source | Example |
|---|---|---|
| 1 | `--memory` | `--memory 96Gi`, or `--memory auto` to size it |
| 2 | `memory:` in the model's catalog entry | `memory: 128Gi`, or `memory: auto` to size it |
| 3 | The category default | `16Gi` |

**KV cache and context**, depending on what is set:

| Memory | `VLLM_CPU_KVCACHE_SPACE` | Result |
|---|---|---|
| `auto` | not set | KV for 16 streams of `max_model_len` tokens; memory sized around it; `--max-model-len` emitted |
| `auto` | set | Rejected: `memory: auto` sizes both. Set `memory` instead, or remove the KV setting |
| set, or the category default | set, or the category `kv_cache_gb` | Used as is; the [pre-flight](#pre-flight-checks) checks that they fit |

- **Where the KV setting counts as set:** in the model's `env:`, a version's `env:`, or
  `--env`. The category's `${KV_CACHE_GB}` placeholder does not count.
- **`max_model_len`** is `--max-model-len` from the model's `args:` or `--arg`. Otherwise
  it is 8192, capped at the model's own limit (`max_position_embeddings`, `n_positions`,
  `seq_length` or `max_sequence_length`; `max_target_positions` for Whisper). With
  `--max-model-len auto` (or `-1`) the KV cache is sized for 8192 tokens and vLLM fits the
  context to it. Values take vLLM's suffixes: `32K` = 32,768, `25.6k` = 25,600. A given
  value is never emitted twice.
- **Engine args** are those vLLM receives, later ones winning: category, version and model
  `args`, `--arg` / `--tp`, and on OpenShift the `VLLM_ADDITIONAL_ARGS` env, which its
  template appends after them.
- **Which runtimes:** vLLM on CPU, i.e. a runtime with no `container:` (KServe's vLLM
  command) and `device: cpu` (the default): `vllm`, and runtimes built the same way. GPU
  runtimes and OpenVINO are not sized; `memory: auto` on them is rejected.

## The formula

```
memory_gi   = ceil( (W + TP × kv_gib + TP × 3 GiB) × 1.15 )

kv_gib      = max( 2, ceil(per_request / GiB), ceil(per_request × streams / GiB) )
streams     = min(16, --max-num-seqs)
per_request = KV-pool bytes one full-length request holds on one worker (below)
```

| Term | Meaning |
|---|---|
| `W` | Weight bytes: the size of the repo-root `.safetensors` files (subfolders such as `original/` and `consolidated.*` hold other formats). For quantized checkpoints the file size is the resident size; an unquantized checkpoint served with a wider `--dtype` (bf16 as float32) is scaled up. A draft model's weights are added |
| `TP` | The last `--tensor-parallel-size` of the model's `args` (category, version, model), then `--arg` / `--tp`, as vLLM receives it: one vLLM worker per rank |
| `kv_gib` | `VLLM_CPU_KVCACHE_SPACE`. vLLM allocates it **on every worker**, so the pod holds `TP ×` it |
| `3 GiB` | Per-worker runtime: Python, PyTorch, oneDNN, prefill activations, sampling buffers |
| `1.15` | Headroom for fragmentation, load-time page cache, and estimate error |

### Tensor parallelism

Weights are sharded, not copied, so `W` appears once. KV heads are split across workers,
so the per-worker KV shrinks with TP while the pod total stays the same. Only the
per-worker overhead grows. Example: Llama-3.1-8B, 8K context, 16 streams:

| TP | KV heads per worker | `VLLM_CPU_KVCACHE_SPACE` | Memory |
|---|---|---|---|
| 1 | 8 | 16 | 40Gi |
| 2 | 4 | 8 | 43Gi |
| 4 | 2 | 4 | 50Gi |

When TP exceeds the KV head count, each worker still holds one head: heads are
replicated, so the per-worker KV stops shrinking. Deploys whose TP does not divide the
attention heads (or that neither divides nor is divided by the KV heads) are rejected
before anything is applied, as vLLM would fail at start-up.

## Layouts

Each layer holds KV in a shared pool of fixed-size pages. This follows the CPU backend
of vLLM 0.28 (`vllm/v1/core/kv_cache_utils.py`, `vllm/v1/kv_cache_interface.py`).

```
bytes per token, one layer, one worker = 2 (K and V) × ceil(kv_heads / TP) × head_dim × dtype bytes
page = widest layer's bytes per token × block size (128); narrower layers fit more tokens per page
```

| Layer type | Tokens one request holds | Config fields read |
|---|---|---|
| Full attention | `max_model_len` | `num_key_value_heads` (else `num_attention_heads`), `head_dim` (else `hidden_size / num_attention_heads`) |
| Sliding window | `min(sliding_window − 1 + --max-num-batched-tokens, max_model_len)` + 1 page | `layer_types: sliding_attention`, `sliding_window` |
| Linear attention (Gated DeltaNet) | a fixed state: 1 page, 2 with prefix caching | `layer_types: linear_attention`, `linear_num_key_heads`, `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_conv_kernel_dim` |
| Cross-attention (encoder-decoder) | `max(max_source_positions, --max-num-batched-tokens)`: vLLM reserves the encoder budget | `is_encoder_decoder`, `decoder_layers`, `decoder_attention_heads`, `d_model`, `max_source_positions` |

Layers of one kind (full, each sliding window, cross, linear; a draft model's full-attention
layers join the model's) form KV cache groups of equal size: the smallest kind's layer count,
or the largest's when it is under 1.5× the smallest. A kind whose count is not a multiple of
it is padded with layers that cost as much as its own, e.g. 8 full + 9 sliding is allocated
as 9 + 9. The validated layouts below need no padding.

How the layout is chosen:
1. **Encoder-decoder** (`is_encoder_decoder: true`, e.g. Whisper): every decoder layer has
   self-attention, capped at `max_target_positions`, plus cross-attention over the audio
   frames.
2. **`layer_types` present:** one layer per entry. Fields under `text_config` are used
   when the config has one (multimodal models).
3. **Otherwise:** every one of `num_hidden_layers` is full attention. A model-wide
   `sliding_window` without `layer_types` is counted as full attention, which oversizes
   safely.

Engine args that change the result: `--dtype`, where `auto` means the checkpoint's
`torch_dtype` and float32 doubles the bytes; `--kv-cache-dtype fp8*`, which halves them;
`--block-size`; `--max-num-batched-tokens`; `--max-num-seqs`; `--enable-prefix-caching`;
and `--max-model-len`.

### Dense: `Qwen/Qwen3-4B-Instruct-2507`

```
F F F F … F    36 full layers, num_key_value_heads 8, head_dim 128, bfloat16
per token:   36 × 2 × 8 × 128 × 2 B = 144 KiB
KV:          144 KiB × 8192 × 16 = 18 GiB
memory:      (7.49 + 18 + 3) × 1.15 → 33Gi
```

### Sliding window: `google/gemma-4-26B-A4B-it`

```
S S S S S F S S S S S F S S S S S F S S S S S F S S S S S F    25 sliding + 5 full
sliding: num_key_value_heads 8 × head_dim 256      → 8 KiB/token, window 1024
full:    num_global_key_value_heads 2 × global_head_dim 512 → 4 KiB/token
```

Gemma 4's full layers use the global head shape. With `attention_k_eq_v` they also use
the global KV head count. vLLM still stores both K and V.

| Layers | Tokens held | Per request |
|---|---|---|
| 5 full | 8192 | 160 MiB |
| 25 sliding | 1023 + 2048 + 128 = 3200 | 625 MiB |

KV = 785 MiB × 16 → 13 GiB. Memory = (48.07 + 13 + 3) × 1.15 → **74Gi**. Treating all
30 layers as full would give 30 GiB of KV and 93Gi.

### Hybrid linear attention: `Qwen/Qwen3.6-35B-A3B`

```
L L L F L L L F … L L L F    30 linear + 10 full
full:   num_key_value_heads 2 × head_dim 256 → 2 KiB/token
linear: recurrent state 32 × 128 × 128 × 4 B (fp32 on AMX CPUs) + conv state (2·16·128 + 32·128) × 3 × 2 B
        ≈ 2.05 MiB per request
```

vLLM raises the attention block until a page also holds one linear state: 9 × 128 =
1152 tokens, so each page is 2.25 MiB. With the catalog's 32K context and prefix caching
off, a request holds `10 × ceil(32768 / 1152) + 30 × 1 = 320` pages = 720 MiB. KV =
720 MiB × 16 → 12 GiB, and memory → **95Gi**.

### Encoder-decoder: `openai/whisper-small`

```
decoder layer × 12: self-attention ≤ 448 tokens + cross-attention over 1500 audio frames
heads 12, head_dim 768 / 12 = 64, float32 (the catalog serves it with --dtype float32)
per request: 12 × (4 + 16 pages) × 128 × 6 KiB = 180 MiB   (cross-attention: the 2048 batched tokens)
```

KV → 3 GiB, `--max-model-len 448`, memory → **8Gi**.

### Speculative decoding: a draft model

A `draft_model_id`'s layers draw from the same pool, so its layout is added to the main
model's and its weights to `W`. For example, Llama-3.1-8B with a Llama-3.2-1B draft:
KV (128 + 32) KiB × 8192 × 16 = 20 GiB, weights 14.96 + 2.30 GiB, memory → **47Gi**.

## Pre-flight checks

These run before the weights download, for every CPU policy:

| Check | Fails when | Result |
|---|---|---|
| Topology | TP does not split the attention / KV heads | Abort (also with `--dry-run`), with or without `memory: auto` |
| Fit | A set `memory` is below weights + TP × KV + TP × 3 GiB (no headroom) | Abort with the numbers |
| NUMA | One TP worker's share, weights ÷ TP + KV + 3 GiB, exceeds one NUMA node's memory | Abort, suggesting the smallest `--tp` that fits (up to the node's `nri.intel.com/max-tp`), or a smaller KV / context when none does. TP is never changed for you |
| Placement | No node the model can use has the pod's CPU **and** memory free (allocatable minus running pods' requests) | Abort; the pod would stay Pending. This is the CPU-policy pre-flight, which gets the sized memory |

The NUMA check exists because vLLM binds each TP worker's memory to the NUMA node of its CPUs (MEMBIND), so a worker cannot spill onto another node: past one NUMA node it fails at load even when the pod's limit is larger. A NUMA node's size is the node's memory capacity ÷ its `nri.intel.com/numa-nodes` label, read per node at deploy time, so it follows each host's hardware; if any node the model can use lacks the labels (no NRI), the check is skipped, since that node may fit the worker. NUMA nodes of one host are assumed equal in size, which holds for Xeon SNC layouts but not for uneven DIMMs or CPU-less (CXL) memory nodes.

The fit check matters most for TP. A set memory does not grow with `--tp`, but each
worker allocates the full `VLLM_CPU_KVCACHE_SPACE`. Without the check, `--tp 2` on a
model with `memory: 64Gi` and a 40 GiB KV cache would be OOM-killed at load.

`MM_PREFLIGHT=warn` (or the legacy `MM_NRI_PREFLIGHT=warn`) reports a failure and carries on.
`MM_SKIP_PREFLIGHT=1` skips the fit and NUMA checks; see [CPU Pinning & NUMA](cpu_pinning.md#pre-flight-capacity-check) for the placement check.

## When a model cannot be sized

`memory: auto` stops the deploy, before anything is applied, with the reason:

| Situation | Reason |
|---|---|
| Hugging Face unreachable, or a gated repo (Llama, Gemma) without `HF_TOKEN` | `config.json not readable (gated repo without HF_TOKEN?)` |
| No safetensors files (`.bin` only), for the model or its draft | `no safetensors size on HuggingFace` / `… for the draft model` |
| No context length in `config.json` | `context length unknown` |
| MLA (`kv_lora_rank`) or state-space (`state_size`) attention, or an unknown `layer_types` entry | `MLA or state-space attention` / `layer type …` |
| `--pipeline-parallel-size` or `--data-parallel-size` > 1 | `--pipeline-parallel-size > 1` |

Set `memory:` (and `VLLM_CPU_KVCACHE_SPACE`) for such a model instead.

A sized deploy records what it was sized from (a hash of the model, draft, image, TP and
effective engine args) and the sized `--max-model-len` as annotations on the
`LLMInferenceService` (`memory-sizing.model-manager.io/inputs`, `…/max-model-len`). A
re-deploy of a catalog `memory: auto` model with the same inputs keeps the live memory, KV
cache and context, as `cpu: auto` keeps its CPU: an Ansible re-run then neither calls
Hugging Face nor changes a running model. Any changed input (e.g. `--tp 2`, another
`--max-model-len`, a new image), a model deployed without the annotation, or
`--memory auto` on the command line sizes it again.

Hugging Face is asked for two documents per model (the file list with sizes, and
`config.json`); after a connection failure the deploy stops asking.

Without `memory: auto`, the set or default values are checked against the weights, whose
size is public even for gated repos: the deploy warns when they cannot hold weights +
TP × (KV + 3 GiB) (`Memory 16Gi of llama-3-1-8b is below the 26.0 GiB its weights 15.0 +
KV 1 x 8 + overhead 1 x 3 need.`), and the memory pre-flight then stops it. When the weight
size is unknown, nothing is checked. These values do not change with `--tp`.

## Tuning

| Goal | Change |
|---|---|
| Less memory | `--arg --max-model-len=4096` with `memory: auto` |
| Longer context | `--arg --max-model-len=32768`: KV grows linearly with it |
| More concurrent long requests | Set `memory:` and `VLLM_CPU_KVCACHE_SPACE` (in `env:`) instead of `memory: auto`; sized KV stops at 16 full-length streams |
| A known-good fixed setup | Set both `memory:` and `VLLM_CPU_KVCACHE_SPACE` in the model's `env:` |

## Limits

- The constants (3 GiB overhead per worker, 1.15 headroom, 16 streams, 8192 tokens) are
  estimates. Calibrate them against the peak RSS of real deploys.
- The KV pool rules are those of vLLM 0.28's CPU backend. Recheck them when the
  catalog's `default_version` moves, especially the hybrid page alignment and the
  sliding-window token count.
- Sizing reads `main` of the Hugging Face repo, the revision the download Job pulls.
