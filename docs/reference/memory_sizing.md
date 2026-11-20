# Memory & KV-Cache Sizing

`memory: auto` makes `model-manager deploy` size a vLLM model's pod memory, KV cache and
context from the model itself: its Hugging Face `config.json` (layer layout), its
safetensors sizes (weights) and the engine args it is served with. Only those small JSON
documents are fetched, before the weight download.

## When it applies

| Deploy | Sized? |
|---|---|
| Catalog model with `memory: auto`, or any deploy with `--memory auto` | Yes. Stops with the reason if it can't size the model ([below](#when-a-model-cannot-be-sized)) |
| `--hfid` without `--memory`, category `llm` | Yes, by default. Falls back to the category memory, with a warning, if it can't size the model |
| Catalog model with a set `memory` (or none) | No: its `memory` and KV are used as is, and only [checked to fit](#pre-flight-checks) |

Only **vLLM on CPU** (no runtime `container:`, `device: cpu`) and categories **`llm`** and
**`vlm`** are sized. `memory: auto` on OpenVINO, GPU runtimes, `embed` or `rerank` is rejected.

> [!WARNING]
> A **gated** repo's `config.json` is gated too (Llama, Gemma, Mistral). Set `HF_TOKEN`,
> or `memory: auto` stops with `config.json not readable (gated repo without HF_TOKEN?)`.

Sizing sets three things in the manifest, and prints how it got there:

| Setting | Value |
|---|---|
| `resources.requests.memory` = `limits.memory` | The pod's memory |
| `VLLM_CPU_KVCACHE_SPACE` | The KV cache, in GiB, **per vLLM worker** |
| `--max-model-len` | The longest request vLLM accepts: the model's own context, unless you set one |

```
Memory: 39Gi (auto) = weights 15.0 + overhead 3 (1 x 3) + headroom 2.7 (15%) + KV 16 (1 x 16) + margin 2 GiB | KV for up to 1 x 131072 tokens
```

With load allowances, the log names each one:

```
Memory: 52Gi (auto) = weights 7.0 + load 29.0 (4.0 image encoder + 25.0 video profiling at 0.2 per 1K of 128000 tokens) + overhead 3 (1 x 3) + headroom 5.8 (15%) + KV 5 (1 x 5) + margin 2 GiB | KV for up to 1 x 128000 tokens
```

## Defaults

**By default the KV cache holds one request of the model's full context.**

| Parameter | Default | Override |
|---|---|---|
| Context (`--max-model-len`): sized for, and served | **The model's own limit** from `config.json` (`max_position_embeddings`, …) | `--max-model-len` in `args` / `--arg` |
| `kv_streams`: requests of that context the KV holds at once | **1** | `kv_streams:` / `--kv-streams` |
| Runtime overhead | **3 GiB per TP worker** | [`sizing:`](#tuning-the-constants) `overhead_gib` |
| Headroom | **× 1.15** on weights + load + runtime, **not on the KV**; plus a fixed **2 GiB** | `sizing:` `headroom`, `margin_gib` |
| Minimum KV | **2 GiB** per worker | `sizing:` `kv_min_gib` |
| Load allowances | **0.4 × weights** if quantized; **+4 GiB** with vision/audio; **+0.2 GiB per 1K tokens** of context for native video (Qwen-VL); per-layer embeddings in fp32 (Gemma E-series) | `--limit-mm-per-prompt` set to 0; `sizing:` `quantized_repack`, `multimodal_gib`, `video_gib_per_1k` |
| TP | The last `--tensor-parallel-size` vLLM receives, else 1 | `--tensor-parallel-size` in `args` / `--tp` |

## The formula

```
memory_gi = ceil( (W + load + TP × 3 GiB) × 1.15  +  TP × kv_gib  +  2 GiB )
kv_gib    = max( 2, ceil(per_request × kv_streams / GiB) )      # per worker; default: one full-context request
```

### Why these constants

A vLLM CPU pod runs three processes. Measured live (non-reclaimable memory, idle):

| Pod (W, KV) | API server | EngineCore | Worker | Worker − W − KV | Pod − W − KV |
|---|---|---|---|---|---|
| Qwen3-0.6B (1.4, 5 GiB) | 0.90 | 0.58 | 7.29 | 0.9 | 2.4 |
| Qwen3-4B (7.5, 36 GiB) | 0.90 | 0.58 | 45.50 | 2.0 | 3.5 |
| Llama-3.1-8B (15.0, 16 GiB) | 0.93 | 0.71 | 34.24 | 3.2 | 4.9 |

- **API server** (`vllm serve`): HTTP front end, tokenizer, detokenizer. One per pod.
- **EngineCore**: the scheduler. It holds the block tables but no KV data. One per pod.
- **Worker** (`VLLM::Worker`): one per TP rank. It holds that rank's share of the weights,
  the whole `VLLM_CPU_KVCACHE_SPACE` pool, and its own runtime.

| Constant | Value | Why |
|---|---|---|
| Runtime overhead | **3 GiB per TP worker** | Each worker is a separate Python process. It carries its own torch / oneDNN libraries, kernel scratch buffers, and the activations of one forward pass of up to `--max-num-batched-tokens` tokens (hidden states, MLP intermediates, logits). It is 0.9–3.2 GiB on the models above. The front end adds about 1.5 GiB once per pod, so at TP 1 the 3 GiB covers both for small and mid-size models. At TP > 1 the front end is counted again for each worker, which errs on the safe side |
| Headroom | **× 1.15**, on weights + load + runtime only | These parts are estimates that grow with the model. Activations scale with the hidden size and vocabulary, so the runtime outside W + KV goes from 2.4 GiB (0.6B) to 4.9 GiB (8B) to 8–9 GiB (30–35B). 15% of the weights covers that growth: for Llama-8B, 3 + 0.15 × 18 = 5.7 GiB against 4.9 measured |
| No headroom on the KV | — | vLLM allocates the KV pool once, at exactly `VLLM_CPU_KVCACHE_SPACE`, and requests take 128-token blocks inside it, so it never grows. Measured live, the memory outside W + KV stayed the same as the KV grew 14×: Qwen3-0.6B 2.4 → 2.5 GiB (KV 5 → 70), Qwen3-4B 3.4 → 3.5 GiB (KV 36 → 144). Under 16 concurrent requests it rose only 0.1–0.2 GiB. A 15% on the KV would buy nothing and grows with `kv_streams`: Llama-3.1-8B × 16 would be 316Gi instead of 279Gi |
| Margin | **+2 GiB per pod** | It is not tied to the KV. It covers what a percentage misses on small models: 15% of Qwen3-0.6B's 4.4 GiB is only 0.7 GiB, while the runtime differs by ±1 GiB between models and vLLM versions. It also leaves room for the page cache that reading the safetensors leaves in the cgroup (3.8 GiB on Qwen3-4B). That cache is reclaimable, but if non-reclaimable memory sits right at the limit, the kernel has to reclaim it while serving |
| Minimum KV | **2 GiB per worker** | vLLM refuses to start if the pool cannot hold one `--max-model-len` request. A short-context model's pool would also be tiny: Llama-3-8B-AWQ (8K) needs 1 GiB for one request. 2 GiB lets it serve 2 × 8K at once. It only applies when one full-context request needs less than 2 GiB (8K-context models, Whisper) |

| Term | Meaning |
|---|---|
| `W` | The repo-root `.safetensors` files (`original/`, `consolidated.*` skipped), plus a draft model's. Unquantized weights served with a wider `--dtype` scale up (bf16 as float32: × 2) |
| `load` | The [load allowances](#load-allowances) |
| `TP` | One vLLM worker per rank. Weights are split across them; each worker allocates the full `kv_gib` and its own 3 GiB |
| `per_request` | The KV bytes one request of the full context holds on one worker, by [layer layout](#how-the-kv-cache-is-computed) |

KV heads are split across workers, so the KV per worker shrinks with TP while the pod total
stays the same. Example: Llama-3.1-8B, 1 × 131,072 tokens:

| TP | `VLLM_CPU_KVCACHE_SPACE` | Memory |
|---|---|---|
| 1 | 16 | 39Gi |
| 2 | 8 | 43Gi |
| 4 | 4 | 50Gi |

## Sizing vs vLLM serving

Sizing builds a KV pool of a fixed size. vLLM shares that pool between whatever requests are
running, a 128-token block at a time, as their tokens arrive.

| Setting | Means | Set by |
|---|---|---|
| `kv_streams` × context | **Sizing:** how big the pool is. The guarantee that `kv_streams` requests of the full context fit at once | `memory: auto` |
| `--max-model-len` | **Serving:** the longest single request accepted. Longer requests get HTTP 400 (`This model's maximum context length is …`) before they run, however much KV is free | Sizing (the model's context), or you |
| `--max-num-seqs` | **Serving:** the most requests vLLM batches at once. The catalog's 256 means "no practical limit". **Not a sizing target** | Catalog args |

- **One full-context request, or many shorter ones.** The default pool holds the model's whole
  context once, e.g. 131,072 tokens for Llama-3.1-8B. That's also 16 × 8K, or 64 × 2K, so
  short requests still run together.
- **How many run at once** is `min(--max-num-seqs, pool tokens ÷ tokens per request)`.
- **When the pool is full**, new requests wait and running ones may be preempted (evicted and
  recomputed later). They get slower but don't fail.
- **Short-context models get a small pool.** Llama-3-8B-AWQ (8K) gets the 2 GiB floor, about
  2 × 8K. Raise `kv_streams` where many users need the full context at once.
- **Long prefills are slow on CPU:** 120K tokens on Qwen3-4B took about 15 minutes.

Example: Qwen3-4B, defaults. The pool is 36 GiB = 262,144 tokens, its full context:

| Request | Result |
|---|---|
| 32 × 8K at once | All run together |
| 1 × 256K | Runs, holding the whole pool |
| 2 × 128K at once | Both run |
| 3 × 128K at once | 2 run, 1 waits until a request finishes |
| 1 request over 262,144 tokens | HTTP 400 immediately |

## Overrides

| Goal | CLI | Catalog | Llama-3.1-8B result |
|---|---|---|---|
| Defaults | `--memory auto` | `memory: auto` | 39Gi, KV 16: 1 × 131,072 |
| More full-context requests at once | `--kv-streams 4` | `kv_streams: 4` | 87Gi, KV 64 |
| Shorter context, less memory | `--arg --max-model-len=32768` | `"--max-model-len", "32768"` in `args` | 27Gi, KV 4: 1 × 32K |
| Many guaranteed short requests | `--arg --max-model-len=8192 --kv-streams 16` | both of the above | 39Gi, KV 16: 16 × 8K, serves 8192 |
| Split across NUMA nodes | `--tp 2` | `"--tensor-parallel-size", "2"` | 43Gi, KV 8 per worker |
| Multimodal model without video | `--arg --limit-mm-per-prompt.video=0` | in `args` | Qwen2.5-VL-3B: 52Gi → 24Gi |
| Multimodal model as text only | `--arg --limit-mm-per-prompt.image=0 --arg --limit-mm-per-prompt.video=0` | in `args` | Drops the multimodal allowances |
| Recalibrate for other hardware or a new vLLM | — | [`runtimes.vllm.sizing`](#tuning-the-constants) | Every `memory: auto` model of the runtime |
| Fixed memory and KV, no sizing | `--memory 64Gi --env VLLM_CPU_KVCACHE_SPACE=40` | `memory: 64Gi`, `VLLM_CPU_KVCACHE_SPACE: "40"` in `env` | Used as is, fit-checked |

```yaml
# models.yaml
- name: llama-3-1-8b
  model_id: meta-llama/Llama-3.1-8B-Instruct
  memory: auto
  kv_streams: 4
  args: [ "--max-model-len", "32768", "--tensor-parallel-size", "1" ]   # KV for 4 x 32K
```

Rules:

- **Precedence:** the CLI wins over the catalog. Engine args are read as vLLM gets them, the
  last one winning: category, version and model `args`, then `--arg` / `--tp`, then (on
  OpenShift) `VLLM_ADDITIONAL_ARGS`. vLLM's spellings and suffixes are accepted: `--max_model_len`,
  `32K` = 32,768, `25.6k` = 25,600.
- **`--max-model-len auto` (or `-1`)** sizes for the model's context and lets vLLM fit it to the pool.
- **The context is capped** by a draft model's own limit, and by `--max-num-batched-tokens` when
  chunked prefill is off (`--no-enable-chunked-prefill`).
- **`memory: auto` + `VLLM_CPU_KVCACHE_SPACE`** (in the model's or a version's `env:`, or `--env`)
  is rejected, because `memory: auto` sizes both. The category's `${KV_CACHE_GB}` doesn't count.
  For the `--hfid` default, a KV setting keeps the category memory instead.
- **`kv_streams`** has no upper limit: the [pre-flight](#pre-flight-checks) catches a value
  that doesn't fit. It warns, and has no effect, without `memory: auto`. It also warns
  above `--max-num-seqs`, because vLLM can't use the extra KV.

## Pre-flight checks

These run before the weights download. Only the topology check also runs with `--dry-run`.

| Check | Fails when | Result |
|---|---|---|
| Topology | TP doesn't split the attention heads, or neither divides nor is divided by the KV heads | Abort |
| Fit | A set `memory` < weights + load + TP × (KV + 3 GiB), without headroom | Abort, with the numbers |
| NUMA | One worker's share, (weights + load) ÷ TP + KV + 3 GiB, exceeds one NUMA node | Abort, suggesting the smallest `--tp` that fits (with the KV at that TP), or a lower `kv_streams` / KV / context. TP is never changed for you |
| Placement | No usable node has the pod's CPU and memory free | Abort: the pod would stay Pending ([CPU Pinning & NUMA](cpu_pinning.md#pre-flight-capacity-check)) |

- **Why the NUMA check:** vLLM binds each worker's memory to its NUMA node (MEMBIND), so a
  worker fails at load past one node, even under the pod limit.
- **How the NUMA size is read:** node memory ÷ the `nri.intel.com/numa-nodes` label, per node.
  The check is skipped if any usable node lacks the label. NUMA nodes of a host are assumed
  equal, which is true of Xeon SNC but not of uneven DIMMs or CXL memory.
- **Example:** Llama-3.1-8B with `--kv-streams 16` (16 × 131K) needs 274 GiB per worker at
  TP 1. On 252 GiB NUMA nodes it aborts with `Deploy with --tp 2 … 128 GiB of KV`.
- **Overrides:** `MM_PREFLIGHT=warn` reports and continues. `MM_SKIP_PREFLIGHT=1` skips the
  fit and NUMA checks.

## When a model cannot be sized

`memory: auto` stops before anything is applied. The `--hfid` default warns with the same
reason and uses the category memory instead.

| Situation | Reason given |
|---|---|
| Gated repo without `HF_TOKEN`, or Hugging Face unreachable | `config.json not readable …` / `Hugging Face is unreachable (proxy: …)` |
| No safetensors (`.bin` only), for the model or its draft | `no safetensors size …` |
| No context length in `config.json` | `context length unknown` |
| State-space attention (`state_size`) or an unknown `layer_types` entry | `state-space attention` / `layer type …` |
| `--pipeline-parallel-size` or `--data-parallel-size` > 1 | `--pipeline-parallel-size > 1` |

Set `memory` and `VLLM_CPU_KVCACHE_SPACE` for such a model instead.

Without `memory: auto`, a set or default memory is still checked against the weights,
whose size is public even for gated repos: `Memory 16Gi of llama-3-1-8b is below the 26.0
GiB its weights 15.0 + KV 1 x 8 + overhead 1 x 3 need.`

## Redeploys

A sized deploy records its inputs as annotations on the `LLMInferenceService`:
`memory-sizing.model-manager.io/inputs` is a hash of the model, draft, image, TP, effective
args, `kv_streams` and the [`sizing:` overrides](#tuning-the-constants) (each when set), and
`…/max-model-len` is the sized context.

- **Same inputs:** the redeploy keeps the live memory, KV and context, without calling Hugging
  Face, so an Ansible re-run never changes a running model.
- **Changed inputs** (e.g. `--tp 2`, other args, a new image, `--kv-streams`, a `sizing:` value): it's sized again.
- **`--memory auto` on the command line:** always sizes again.

Hugging Face is asked for two documents per model (the file list and `config.json`). After
a connection failure, it isn't asked again.

## How the KV cache is computed

This follows vLLM 0.28's CPU KV allocator (`vllm/v1/core/kv_cache_utils.py`).

```
bytes per token, one layer, one worker = 2 (K and V) × ceil(kv_heads / TP) × head_dim × dtype bytes
page = 128 tokens (--block-size) of the widest layer; narrower layers fit more tokens per page
```

| Layer type | Tokens one request holds | Config fields |
|---|---|---|
| Full attention | the context | `num_key_value_heads`, `head_dim` (else `hidden_size / num_attention_heads`) |
| Sliding window | `min(window − 1 + --max-num-batched-tokens, context)` + 1 page | `layer_types: sliding_attention`, `sliding_window` |
| Linear attention (Gated DeltaNet) | a fixed state: 1 page, 2 with prefix caching. The page grows until it holds one state | `layer_types: linear_attention`, `linear_*` |
| MLA (DeepSeek, Kimi) | the context, one `kv_lora_rank + qk_rope_head_dim` latent, not split by TP | `kv_lora_rank`, `qk_rope_head_dim` |
| Cross-attention (encoder-decoder) | `max(max_source_positions, --max-num-batched-tokens)` | `is_encoder_decoder`, `decoder_*`, `d_model` |

- **Layout:** encoder-decoder (Whisper) → MLA → `layer_types` → otherwise every
  `num_hidden_layers` is full attention. A model-wide `sliding_window` without `layer_types`
  counts as full, which oversizes safely. `text_config` is used when present (multimodal
  models).
- **Shared-KV layers** (`num_kv_shared_layers`, Gemma 3n / Gemma 4 E2B, E4B) hold none of their own.
- **Group padding:** layer kinds form KV groups of equal size, as in vLLM, so 8 full + 9
  sliding is allocated as 9 + 9.
- **Draft models** draw from the same pool, so their layers add to the model's.
- **Engine args that change it:** `--dtype` (`auto` = the checkpoint's; float32 doubles),
  `--kv-cache-dtype fp8*` (halves), `--block-size`, `--max-num-batched-tokens`,
  `--enable-prefix-caching`, `--max-model-len`.

| Model (defaults: 1 × full context) | Layout | Context | KV / worker | Memory |
|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | 36 full, 144 KiB/token | 262,144 | 36 GiB | 51Gi |
| Llama-3.1-8B + Llama-3.2-1B draft | 32 + 16 full | 131,072 | 20 GiB | 46Gi |
| gemma-4-26B-A4B-it | 25 sliding (window 1024) + 5 full; vision, audio, video | 262,144 | 6 GiB | 72Gi |
| Qwen3.6-35B-A3B (catalog: 32K, no prefix caching) | 30 linear + 10 full, 1152-token pages | 32,768 | 2 GiB | 85Gi |
| whisper-small (float32) | 12 decoder self-attention + 12 cross-attention | 448 | 2 GiB | 9Gi |
| gemma-4-E4B-it | 24 of 42 layers cache (18 share KV); multimodal; per-layer embeddings | 131,072 | 3 GiB | 43Gi |
| Qwen2.5-VL-3B-Instruct (`vlm`) | 36 full; vision + native video | 128,000 | 5 GiB | 52Gi (24Gi without video) |

On a live cluster, vLLM's reported KV capacity was within 1.6% of this calculation for all
of these layouts (at 16 × 8K, and at 1 × 262,144 for Qwen3-4B).

### Load allowances

Memory a deploy needs beyond its weights, KV cache and 3 GiB runtime, from what `config.json`
declares. Each one is memory that vLLM uses at start-up or keeps resident, and that the weights
and KV don't account for. The deploy log names each allowance it adds, e.g.
`load 29.0 (4.0 image encoder + 25.0 video profiling at 0.2 per 1K of 128000 tokens)`.

| Allowance (log name) | When | Amount | What uses the memory | Measured (cgroup peak) |
|---|---|---|---|---|
| `quantized repack` | `quantization_config` (AWQ, GPTQ) | 0.4 × weights | The checkpoint's packed int4 layout isn't the one the CPU kernel runs. At load vLLM converts each weight to the kernel's layout, and the source and repacked tensors are both resident until the source is freed | Llama-3-8B-AWQ: about 0.2 × weights. 0.4 is twice that, as only one quantized model was measured |
| `image encoder`, `audio encoder` | `vision_config` / `audio_config`, unless every modality the model has is limited to 0 | 4 GiB | At start-up vLLM profiles the encoder: it runs the largest dummy image or audio clip `--limit-mm-per-prompt` allows, to size its activations and encoder cache. The ViT / audio-encoder activations of that pass, and the multimodal input processing, are on top of the weights | Images +2.5 GiB (Qwen2.5-VL-3B); images and audio +2.4 GiB (Gemma 4 E4B) |
| `video profiling` | `vision_config.temporal_patch_size` (Qwen2-VL, 2.5-VL, 3-VL, 3.5, 3.6) with `video_token_id`, unless video is limited to 0 | 0.2 GiB per 1K tokens of context | The same profiling, with a dummy video. These encoders turn time into tokens (temporal patches), so vLLM builds the longest video the context can hold. At 256K that's a 256K-token pass through the vision encoder | Qwen2.5-VL-3B: +2.3 GiB at 32K, +26.4 GiB at 128K |
| `per-layer embeddings` | `hidden_size_per_layer_input` (Gemma 3n, Gemma 4 E2B / E4B) | The tables in fp32: vocab × layers × dim × 4 B | These models have an extra embedding table for every layer (262,144 × 42 × 256 on E4B). The allowance assumes they are resident in fp32, twice their bf16 checkpoint size: 10.5 GiB against 5.25 GiB | Gemma 4 E4B: 10.5 GiB allowed, about 9 GiB used |

- **Why video depends on the context and images don't:** an image's token count is bounded
  by its resolution, but a video with temporal patches grows with `--max-model-len`.
  Frame-sampling encoders (Gemma, LLaVA) take a fixed number of frames, so they add next to
  nothing for video, and only their encoder allowance counts.
- **The video rate is the least certain number.** It grew faster than the context: 0.07 GiB
  per 1K at 32K, 0.21 at 128K. 0.2 fits 128K, but beyond 128K it is extrapolated, not
  measured. Qwen3.6-35B-A3B at its 262,144-token default gets 51.2 GiB of video. If you don't
  serve video, set `--limit-mm-per-prompt.video=0`; if you do, check the first deploy's peak.
- **`--limit-mm-per-prompt`** is read in both forms, `.image 0` and `'{"image": 0, "video": 0}'`.
  Without video, a Qwen-VL model needs a small fraction of the memory: Qwen2.5-VL-3B goes
  from 52Gi to 24Gi.
- **Per pod:** the allowances count once per pod, and are split across workers for the NUMA check.

Checked live with `memory: auto`. Each deploy served an image request; peak is the cgroup's,
with non-reclaimable (anon) memory in brackets. The sized column is the current formula; the
peaks were measured at the slightly lower earlier sizing (51 / 26 / 22 / 41 / 37Gi):

| Deploy | Sized | Peak (anon) |
|---|---|---|
| Qwen2.5-VL-3B, images + video, 128K | 52Gi | 38.3 GiB (37.9) |
| Qwen2.5-VL-3B, images + video, 32K | 28Gi | 26.0 GiB (19.9): the rest is page cache, reclaimed at the limit |
| Qwen2.5-VL-3B, images only, 128K | 24Gi | 19.3 GiB (17.4) |
| Gemma 4 E4B, images + audio + video, 131K | 43Gi | 36.4 GiB (32.3) |
| Gemma 4 E4B, text only, 131K | 38Gi | 29.9 GiB (29.5) |

## Tuning the constants

The runtime overhead, quantized repack and encoder allowances depend on the hardware and the
vLLM version; the KV and weight sizes don't. Recalibrate them on other hardware or a new vLLM
if required, per runtime in `models.yaml`:

```yaml
runtimes:
  vllm:
    sizing:                   # every key optional; unset = this default
      overhead_gib: 3         # runtime per TP worker
      headroom: 1.15          # on weights + load + runtime (not the KV); >= 1
      margin_gib: 2           # per pod
      kv_min_gib: 2           # KV floor per worker; a whole number >= 1
      quantized_repack: 0.4   # x weights (AWQ, GPTQ)
      multimodal_gib: 4       # image / audio encoder
      video_gib_per_1k: 0.2   # native video, per 1K tokens of context
```

- **Where:** `runtimes.<name>.sizing` in `models.yaml`, or in `runtimes/<name>/values.yaml`
  (`models.yaml` wins, key by key). These numbers depend on the hardware and vLLM, not on the
  model, so they're per runtime. A model's own needs go through the [overrides](#overrides).
- **Checked at load:** an unknown key, a negative value, `headroom` below 1 or a fractional
  `kv_min_gib` stops the deploy with the reason.
- **Shown:** the deploy log ends with the values that differ from the defaults:
  `| runtimes.vllm.sizing: overhead_gib 4, headroom 1.1`.
- **Redeploys:** changed values join the [recorded inputs](#redeploys), so each `memory: auto`
  model is sized again on its next deploy. Values equal to the defaults change nothing.
- **They apply to the pre-flight too:** the fit and NUMA checks use the same `overhead_gib`, and
  the load allowances count against a fixed `memory`.
- **How to measure:** deploy with `memory: auto` and read the pod cgroup's non-reclaimable memory
  (`anon` in `memory.stat`) idle and under load. Page cache (`file`) is reclaimable. Then
  compare: the runtime outside W + KV against `overhead_gib` (+ the headroom on the weights),
  and a multimodal model's extra against its allowance.

## Limits

- **The constants are estimates:** 3 GiB per worker, × 1.15 and 2 GiB, the one-request default and the
  allowances. [Tune them](#tuning-the-constants) against the peak memory of real deploys. Text models peaked at
  81–93% of the sized limit, with 3–8 GiB of overhead per worker. The multimodal allowances
  come from two models: Qwen2.5-VL-3B and Gemma 4 E4B. Check a large VLM's peak before relying
  on them.
- **The KV rules are vLLM 0.28's CPU backend.** Recheck them when the catalog's
  `default_version` moves. MLA follows vLLM's spec, but is not validated: vLLM 0.28 has no
  CPU MLA backend.
- **The Hugging Face revision:** sizing reads `main` of the repo, the revision the download Job pulls.
