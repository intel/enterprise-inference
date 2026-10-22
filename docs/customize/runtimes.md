# Runtimes

## What is it?

A **runtime** is the engine that actually serves a model. Two are supported:

| Runtime | Use case | Key optimizations |
|---------|----------|-------------------|
| **vLLM** | LLMs, VLMs | NUMA auto-bind, AMX kernels, prefix caching, chunked prefill |
| **OpenVINO (OVMS)** | Embedding, reranking, INT4/INT8 LLMs | OpenVINO IR, MediaPipe graph |

Each engine is configured **once**, in the `runtimes:` section at the bottom of
`models.yaml`. That is where you define the available versions and their container
images, the settings shared by every model on that engine, and the per-workload
(category) defaults — CPU, memory, engine flags, and env vars. Keeping it in one place
means the full picture of what a server runs with is easy to see and to change.

Models then opt into a runtime and version through their
[server binding](catalog.md#choose-which-server-and-version-a-model-may-use).

## How settings layer

Every layer overrides the one before it:

```
env   = runtime common env  +  category env  +  version env delta  +  per-model env  +  --env
args  = category args       +  version args delta  +  per-model args  +  --arg
image = the selected version's image                 (or per-model `image:`)
```

Version selection, highest precedence first:

```
--server-version   >   the model's server_version:   >   the runtime's default_version
```

## vLLM

A **versioned** runtime. Adding a new engine version is a new entry under `versions:` —
no code changes.

```yaml
runtimes:
  vllm:
    default_version: "0.24.0"
    env: { ... }                 # applied to every vLLM pod
    versions:
      "0.24.0":
        image: "docker.io/vllm/vllm-openai-cpu:v0.24.0"
      "0.19.1":
        image: "docker.io/vllm/vllm-openai-cpu:v0.19.1"
        # optional env:/args: DELTAS layer on top of the common env and category args
    categories:
      llm:    { defaults: { cpu: 8,  memory: "16Gi", kv_cache_gb: 8 } }
      embed:  { defaults: { cpu: 6,  memory: "12Gi" } }
      rerank: { defaults: { cpu: 6,  memory: "12Gi" } }
      vlm:    { defaults: { cpu: 16, memory: "48Gi", kv_cache_gb: 8 } }
```

**Adding a version:** copy an existing entry, change the key and `image`, deploy a model
against it with `--server-version`, then promote it by moving `default_version`.

> [!WARNING]
> Per-version `args:` deltas must suit the models that will use them. MoE-only flags
> such as `--enable-expert-parallel` **crash dense models**.

### CPU KV cache

`VLLM_CPU_KVCACHE_SPACE` is deliberately **not** in the shared `env:` — it is set
per-category from that category's `kv_cache_gb`, so each workload class sizes its CPU KV
cache appropriately.

`embed` and `rerank` do no autoregressive generation, so they set no `kv_cache_gb` and
the variable is omitted entirely — vLLM uses its own default.

### Plumbing

Low-level, stable settings live in `model_manager/runtimes/vllm/values.yaml` and are
merged in at load time. `models.yaml` wins on any key clash.

```yaml
device: cpu
kind: LLMInferenceService
routing: epp
```

## OpenVINO (OVMS)

A **flat** runtime — there is no version axis. `server_version` does not apply, and
requesting one with `--server-version` aborts with a clear message.

```yaml
runtimes:
  openvino:
    env: {}
    categories:
      llm:    { defaults: { cpu: 8, memory: "16Gi" } }
      embed:  { defaults: { cpu: 6, memory: "12Gi" } }
      rerank: { defaults: { cpu: 6, memory: "12Gi" } }
```

Plumbing in `model_manager/runtimes/openvino/values.yaml`:

```yaml
image: "openvino/model_server:2026.3"                     # serving image
model_downloader_image: "openvino/model_server:latest-py" # weights download job
device: cpu
kind: InferenceService
format: openvino
runtime_name: openvino-runtime                            # ClusterServingRuntime to reference
routing: aigateway
serving_runtime_manifest: openvino/serving-runtime.yaml   # applied on first deploy
```

### `weight_format` — only for raw Hugging Face repos

`weight_format` is a **conversion** knob for `ovms --pull`, set per category. Use it
(`int8`, `int4`, `fp16`, …) only when `model_id` points at a raw Hugging Face repo that
must be converted and quantized to OpenVINO IR.

> [!IMPORTANT]
> Leave it **unset** for models already published as OpenVINO IR (the `OpenVINO/*`
> repos). Passing `--weight-format` to an already-converted model makes `ovms --pull`
> abort with *"unsupported for OpenVINO models"*.

### API paths

Clients always use the OpenAI `/v1/` paths. For OpenVINO-served models the AI gateway
rewrites `/v1/` → `/v3/` internally, since OVMS uses the v3 API — this is invisible to
callers. See [Inference Request Flow](../reference/request_flow.md).

## Which kind of Kubernetes object?

The runtime's `kind` decides what `model-manager` creates:

| Runtime | Object | Template |
|---|---|---|
| vLLM | `LLMInferenceService` | `model_manager/templates/llm-inference-service.yaml` |
| OpenVINO | `InferenceService` | `model_manager/templates/inference-service.yaml` |

The raw `ClusterServingRuntime` manifests stay as files under
`model_manager/runtimes/<name>/`.
