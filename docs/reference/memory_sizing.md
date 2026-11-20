# Memory & KV-Cache Sizing

## What it does

`memory: auto` in a model's catalog entry, or `--memory auto` on the command line, makes
`model-manager deploy` work out a vLLM chat or vision-language model's (category `llm` or `vlm`) pod memory and KV
cache from the model itself, like `cpu: auto` does for CPUs. It reads:

- the model's `config.json` on Hugging Face, for the layer layout;
- the safetensors index or file sizes, for the weight size;
- the engine args the model is served with.

Sizing is opt-in for catalog models: one without `memory: auto` deploys with its set
`memory` (or the category default) and KV cache, as before; only their
[fit](#pre-flight-checks) is checked. `--hfid` without `--memory` is sized by default, as it
has no catalog memory to keep. When that default cannot size the model, it warns and
deploys with the category memory instead of stopping.

> [!WARNING]
> For a **gated** repo (Llama, Gemma, Mistral) `config.json` is gated too. Without
> `HF_TOKEN`, `memory: auto` cannot size the model and the deploy stops with
> `memory: auto cannot size …: config.json not readable (gated repo without HF_TOKEN?)`.
> Set `HF_TOKEN`, or set `memory:` for that model. (`--hfid` without `--memory` falls back
> to the category memory instead.)

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
| 3 | `--hfid` on vLLM on CPU, category `llm` | `auto`, falling back to 4 when the model cannot be sized |
| 4 | The category default | `16Gi` |

**KV cache and context**, depending on what is set:

| Memory | `VLLM_CPU_KVCACHE_SPACE` | Result |
|---|---|---|
| `auto` | not set | KV for [`kv_streams`](#streams-kv_streams) (default 16) requests of `max_model_len` tokens; memory sized around it; `--max-model-len` emitted |
| `auto` | set | Rejected: `memory: auto` sizes both. Set `memory` instead, or remove the KV setting. `--hfid`'s default `auto` instead gives way to the category memory |
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
  runtimes and OpenVINO are not sized; `memory: auto` on them is rejected, as on `embed` and
  `rerank` models, and `--hfid` on them keeps the category memory.

## The formula

```
memory_gi   = ceil( (W + load + TP × kv_gib + TP × 3 GiB) × 1.15 )

kv_gib      = max( 2, ceil(per_request / GiB), ceil(per_request × streams / GiB) )
streams     = kv_streams if set, else min(16, --max-num-seqs)
per_request = KV-pool bytes one full-length request holds on one worker (below)
```

| Term | Meaning |
|---|---|
| `W` | Weight bytes: the size of the repo-root `.safetensors` files (subfolders such as `original/` and `consolidated.*` hold other formats). For quantized checkpoints the file size is the resident size; an unquantized checkpoint served with a wider `--dtype` (bf16 as float32) is scaled up. A draft model's weights are added |
| `load` | Load-time allowances, see [below](#load-allowances): 0.4 × `W` for a quantized checkpoint, 10 GiB for vision / audio towers, 24 GiB more for video |
| `TP` | The last `--tensor-parallel-size` of the model's `args` (category, version, model), then `--arg` / `--tp`, as vLLM receives it: one vLLM worker per rank |
| `streams` | Full-length requests the KV cache holds at once; see [Streams](#streams-kv_streams) |
| `kv_gib` | `VLLM_CPU_KVCACHE_SPACE`. vLLM allocates it **on every worker**, so the pod holds `TP ×` it |
| `3 GiB` | Per-worker runtime: Python, PyTorch, oneDNN, prefill activations, sampling buffers |
| `1.15` | Headroom for fragmentation, load-time page cache, and estimate error |

### Load allowances

Memory a deploy needs while it loads, beyond its weights, KV cache and runtime. The values
were measured on Xeon in intel/enterprise-inference#86.

| Allowance | When | Amount | Why |
|---|---|---|---|
| Quantized repack | `quantization_config` in `config.json` (AWQ, GPTQ, …) | 0.4 × the weight bytes | vLLM repacks the weights into the CPU kernel's layout at load, both copies briefly resident |
| Multimodal towers | `vision_config` or `audio_config`, unless `--limit-mm-per-prompt` sets every modality the model has to 0 | 10 GiB | vLLM profiles the encoders with their largest input at start-up |
| Video | `video_token_id`, unless `--limit-mm-per-prompt` sets video to 0 | 24 GiB more | A full-size video costs far more to profile than an image (Qwen2.5-VL-3B: ~21 GiB); a minimum, so set `memory` for larger video models |

`--limit-mm-per-prompt` is read in both forms: `--limit-mm-per-prompt.image 0` and
`--limit-mm-per-prompt '{"image": 0, "video": 0}'`. Serving a multimodal model as text only
(image and video 0, plus audio 0 if it has audio) drops both multimodal allowances. The
allowances count once per pod, and are split across the TP workers for the NUMA check.

### Streams: `kv_streams`

`kv_streams` is how many requests of the full `max_model_len` the KV cache holds at the
same time. It is the sizing target for `memory: auto`, and the main knob between memory and
concurrency: the KV cache, and so the memory, grows linearly with it.

| Where | Example | Default |
|---|---|---|
| The model's catalog entry | `kv_streams: 32` | 16 |
| The command line, for one deploy (wins) | `--kv-streams 32` | |

- **No upper limit.** Any value is sized. A value too large for the hardware is caught by the
  [pre-flight](#pre-flight-checks), before the download: the fit check, and the NUMA check,
  which suggests the `--tp` that fits. The KV per worker shrinks as TP splits the KV heads, so
  a larger TP can hold more streams.
- **Without `kv_streams`:** 16, or fewer when `--max-num-seqs` is lower (vLLM would never run
  more at once).
- **Only with `memory: auto`.** With a set `memory`, the KV cache is `VLLM_CPU_KVCACHE_SPACE`
  as given, and `kv_streams` is ignored with a warning.
- **Changing it re-sizes a deployed model.** It is one of the [recorded sizing
  inputs](#when-a-model-cannot-be-sized), when set.

**`kv_streams` is not `--max-num-seqs`.** `--max-num-seqs` is vLLM's scheduler limit: the
most requests it batches at once. The catalog sets it to 256, meaning "no practical limit",
so it is not a sizing target. How many requests actually run together is the smaller of
the two limits:

```
running at once = min( --max-num-seqs,  KV cache tokens / tokens per request )
```

Requests use KV blocks only for the tokens they have so far, so `kv_streams` is the
guarantee for the worst case: that many users each using the whole context at once.
Shorter requests share the same pool, and more of them run together. For example, with
`kv_streams: 16` at 16K tokens, 20 requests of 13K tokens fit. When the pool is full, new
requests wait and vLLM preempts running ones, so they are slower but don't fail.
`kv_streams` above `--max-num-seqs` warns: vLLM can't use the extra KV.

Example: Llama-3.1-8B, 8K context (1 GiB of KV per request at TP 1), on a host whose
NUMA nodes have 252 GiB:

| `kv_streams` | TP | KV per worker | Per worker | Memory | Pre-flight |
|---|---|---|---|---|---|
| 16 (default) | 1 | 16 GiB | 34 GiB | 40Gi | ✅ |
| 32 | 1 | 32 GiB | 50 GiB | 58Gi | ✅ |
| 256 | 1 | 256 GiB | 274 GiB | 316Gi | ❌ NUMA: `Deploy with --tp 2` |
| 256 | 2 | 128 GiB | 138.5 GiB | 319Gi | ✅ |

```bash
./model-manager deploy llama-3-1-8b --memory auto --kv-streams 256 --tp 2
```

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
| MLA (DeepSeek V2/V3, Kimi) | `max_model_len`, one latent of `kv_lora_rank + qk_rope_head_dim` values per token (no separate K and V), on every worker: not split by TP | `kv_lora_rank`, `qk_rope_head_dim` |
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
2. **MLA** (`kv_lora_rank` set): every layer holds one latent per token.
3. **`layer_types` present:** one layer per entry. Fields under `text_config` are used
   when the config has one (multimodal models).
4. **Otherwise:** every one of `num_hidden_layers` is full attention.

The last `num_kv_shared_layers` layers (Gemma 3n, Gemma 4 E2B / E4B) reuse earlier layers' KV
and hold none of their own, so they are left out: Gemma 4 E4B caches 24 of its 42 layers. A model-wide
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

KV = 785 MiB × 16 → 13 GiB. Treating all 30 layers as full would give 30 GiB of KV.
Gemma 4 has vision, audio and video inputs, so the [load allowances](#load-allowances) add
34 GiB: memory = (48.07 + 34 + 13 + 3) × 1.15 → **113Gi**. Served as text only
(`--limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0 --limit-mm-per-prompt.audio 0`)
it is (48.07 + 13 + 3) × 1.15 → 74Gi.

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
| Fit | A set `memory` is below weights + load + TP × KV + TP × 3 GiB (no headroom) | Abort with the numbers |
| NUMA | One TP worker's share, (weights + load) ÷ TP + KV + 3 GiB, exceeds one NUMA node's memory | Abort, suggesting the smallest `--tp` that fits (up to the node's `nri.intel.com/max-tp`), or a lower `kv_streams`, KV or context when none does. TP is never changed for you |
| Placement | No node the model can use has the pod's CPU **and** memory free (allocatable minus running pods' requests) | Abort; the pod would stay Pending. This is the CPU-policy pre-flight, which gets the sized memory |

The NUMA check exists because vLLM binds each TP worker's memory to the NUMA node of its CPUs (MEMBIND), so a worker cannot spill onto another node: past one NUMA node it fails at load even when the pod's limit is larger. A NUMA node's size is the node's memory capacity ÷ its `nri.intel.com/numa-nodes` label, read per node at deploy time, so it follows each host's hardware; if any node the model can use lacks the labels (no NRI), the check is skipped, since that node may fit the worker. NUMA nodes of one host are assumed equal in size, which holds for Xeon SNC layouts but not for uneven DIMMs or CPU-less (CXL) memory nodes.

The fit check matters most for TP. A set memory does not grow with `--tp`, but each
worker allocates the full `VLLM_CPU_KVCACHE_SPACE`. Without the check, `--tp 2` on a
model with `memory: 64Gi` and a 40 GiB KV cache would be OOM-killed at load.

`MM_PREFLIGHT=warn` (or the legacy `MM_NRI_PREFLIGHT=warn`) reports a failure and carries on.
`MM_SKIP_PREFLIGHT=1` skips the fit and NUMA checks; see [CPU Pinning & NUMA](cpu_pinning.md#pre-flight-capacity-check) for the placement check.

## When a model cannot be sized

`memory: auto` stops the deploy, before anything is applied, with the reason. The `--hfid`
default instead warns with the same reason and deploys with the category memory (`Memory:
cannot size qwen3-0-6b: …; using the llm default 16Gi. Set --memory to choose it.`):

| Situation | Reason |
|---|---|
| Hugging Face unreachable, or a gated repo (Llama, Gemma) without `HF_TOKEN` | `config.json not readable (gated repo without HF_TOKEN?)` |
| No safetensors files (`.bin` only), for the model or its draft | `no safetensors size on HuggingFace` / `… for the draft model` |
| No context length in `config.json` | `context length unknown` |
| State-space (`state_size`) attention, or an unknown `layer_types` entry | `state-space attention` / `layer type …` |
| `--pipeline-parallel-size` or `--data-parallel-size` > 1 | `--pipeline-parallel-size > 1` |

Set `memory:` (and `VLLM_CPU_KVCACHE_SPACE`) for such a model instead.

A sized deploy records what it was sized from (a hash of the model, draft, image, TP,
effective engine args and `kv_streams` when set) and the sized `--max-model-len` as annotations on the
`LLMInferenceService` (`memory-sizing.model-manager.io/inputs`, `…/max-model-len`). A
re-deploy of a `memory: auto` model (catalog, or `--hfid` without `--memory`) with the same
inputs keeps the live memory, KV cache and context, as `cpu: auto` keeps its CPU: an
Ansible re-run then neither calls
Hugging Face nor changes a running model. Any changed input (e.g. `--tp 2`, another
`--max-model-len`, a new image), a model deployed without the annotation, or
`--memory auto` on the command line sizes it again.

Hugging Face is asked for two documents per model (the file list with sizes, and
`config.json`); after a connection failure the deploy stops asking.

Without `memory: auto`, the set or default values are checked against the weights, whose
size is public even for gated repos: the deploy warns when they cannot hold weights + load +
TP × (KV + 3 GiB) (`Memory 16Gi of llama-3-1-8b is below the 26.0 GiB its weights 15.0 +
KV 1 x 8 + overhead 1 x 3 need.`), and the memory pre-flight then stops it. When the weight
size is unknown, nothing is checked. These values do not change with `--tp`.

## Tuning

| Goal | Change |
|---|---|
| Less memory | `--arg --max-model-len=4096` with `memory: auto` |
| Longer context | `--arg --max-model-len=32768`: KV grows linearly with it |
| More concurrent long requests | `kv_streams: N` (or `--kv-streams N`) with `memory: auto`; the pre-flight says when N needs a larger `--tp` |
| Less memory, fewer long requests | A lower `kv_streams` |
| A known-good fixed setup | Set both `memory:` and `VLLM_CPU_KVCACHE_SPACE` in the model's `env:` |

## Limits

- The constants (3 GiB overhead per worker, 1.15 headroom, 16 default streams, 8192 tokens, and the
  load allowances) are estimates. MLA follows vLLM's `MLAAttentionSpec`, but vLLM 0.28 has no
  CPU MLA backend, so it is not validated live. Calibrate them against the peak RSS of real deploys.
- The KV pool rules are those of vLLM 0.28's CPU backend. Recheck them when the
  catalog's `default_version` moves, especially the hybrid page alignment and the
  sliding-window token count.
- Sizing reads `main` of the Hugging Face repo, the revision the download Job pulls.
