# Runtimes

## What is it?

A **runtime** is the engine that actually serves a model. Three are supported:

| Runtime | Use case | Key optimizations |
|---------|----------|-------------------|
| **IIMs** (`iims`, the default) | LLMs, VLMs, embedding, reranking | Intel® Inference Microservices images: Intel-validated vLLM profiles per model |
| **vLLM** (`vllm`) | LLMs, VLMs, embedding, reranking | Upstream vLLM CPU images; NUMA auto-bind, AMX kernels, prefix caching, chunked prefill |
| **OpenVINO (OVMS)** (`openvino`) | Embedding, reranking, INT4/INT8 LLMs | OpenVINO IR, MediaPipe graph |

Each engine is configured **once**, in the `runtimes:` section at the bottom of
`models.yaml`. That is where you define the available versions and their container
images, the settings shared by every model on that engine, and the per-workload
(category) defaults — CPU, memory, engine flags, and env vars. Keeping it in one place
means the full picture of what a server runs with is easy to see and to change.

Models then opt into a runtime and version through their
[server binding](catalog.md#choose-which-server-and-version-a-model-may-use).

## How settings combine

Every level overrides the one before it:

```
env   = runtime common env  +  category env  +  version env delta  +  per-model env  +  --env
args  = category args       +  version args delta  +  per-model args  +  --arg
image = the selected version's image                 (or per-model `image:`)
```

Version selection, highest precedence first:

```
--server-version   >   the model's server_version:   >   the runtime's default_version
```

## IIMs

The default runtime. It runs the [Intel® Inference Microservices](https://hub.docker.com/u/intel)
launcher (`python3 -m inference_runtime.cli serve`), which picks a tuned vLLM profile for
the model and then starts vLLM. Catalog models with their own IIM image set it with
`image:` (e.g. `intel/inference-xeon-qwen-qwen3-4b:0.1.0`) and get that model's Intel profile.
Every other model runs on the base image `intel/inference-xeon-base:0.1.0` with its
generic bf16 profile.

```yaml
runtimes:
  iims:
    image: "docker.io/intel/inference-xeon-base:0.1.0"
    categories:
      llm:    { args: [--max-model-len, auto], defaults: { cpu: 8,  memory: "32Gi", kv_cache_gb: 16 } }
      vlm:    { args: [--max-model-len, auto], defaults: { cpu: 16, memory: "48Gi", kv_cache_gb: 16 } }
      embed:  { defaults: { cpu: 6, memory: "8Gi", kv_cache_gb: 2 } }
      rerank: { defaults: { cpu: 6, memory: "8Gi", kv_cache_gb: 2 } }
```

The launcher takes no command-line arguments. The plumbing in
`model_manager/runtimes/iims/values.yaml` sets `container.args_env: INFERENCE_ENGINE_ARGS`, so
the vLLM args (category, model and `--arg`) reach it as one JSON object and are applied
after the profile. `INFERENCE_MODEL_ID` is set from `model_id`, the weights are read from
`/mnt/models`, and `HF_HUB_OFFLINE=1` keeps the launcher off the Hub. Memory is
`weights + kv_cache_gb + headroom`. The launcher does not change CPU placement: vLLM binds
its threads inside the cores the NRI balloon gives the pod.

> [!NOTE]
> Known limits of the 0.1.0 images (vLLM 0.26): `whisper-small` fails its first
> transcription (`pin_memory=True requires a CUDA or other accelerator backend`), so the
> catalog serves it on the vLLM runtime (0.28.0). Gated repos (Llama) need `HF_TOKEN`.

## vLLM

A **versioned** runtime, selected with `--server vllm`. Adding a new engine version is a new entry under `versions:` —
no code changes.

```yaml
runtimes:
  vllm:
    default_version: "0.28.0"
    env: { ... }                 # applied to every vLLM pod
    versions:
      "0.33.0":                  # its KV check counts page cache as used: large models can fail
        image: "docker.io/vllm/vllm-openai-cpu:v0.33.0"
      "0.24.0":
        image: "docker.io/vllm/vllm-openai-cpu:v0.24.0"
      "0.28.0":
        image: "docker.io/vllm/vllm-openai-cpu:v0.28.0"
      "0.19.1":
        image: "docker.io/vllm/vllm-openai-cpu:v0.19.1"
        # optional env:/args: DELTAS on top of the common env and category args
    categories:
      llm:    { defaults: { cpu: 8,  memory: "16Gi", kv_cache_gb: 8 } }
      embed:  { defaults: { cpu: 6,  memory: "16Gi", kv_cache_gb: 16 } }
      rerank: { defaults: { cpu: 6,  memory: "16Gi", kv_cache_gb: 16 } }
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

`embed` and `rerank` do no autoregressive generation, but vLLM CPU still sizes a KV cache
at startup: without the variable it needs most of the NUMA node's memory free, and crashes
when an LLM already runs on that node. So they set it too, from `kv_cache_gb: 16`.

Env values in the runtime, category, version and model layers can use two placeholders, which
are filled in when the manifest is rendered: `${KV_CACHE_GB}` (the category's `kv_cache_gb`) and
`${MODEL_ID}` (the model's Hugging Face `model_id`; the `iims` runtime uses it for
`INFERENCE_MODEL_ID`).

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
image: "openvino/model_server:2026.4.1"   # serving and weights download
device: cpu
kind: LLMInferenceService
routing: epp                              # KServe's scheduler picks a replica per request
container:                                # OVMS replaces KServe's vLLM command
  command: ["/ovms/bin/ovms"]
  args: ["--config_path", "/mnt/models/config_all.json", "--rest_port", "8000", "--port", "9000", "--metrics_enable"]
  ready_path: /v2/models/$model/ready     # Ready once the graph is loaded
  live_path: /v2/health/live
  run_as_user: 5000                       # also owns the downloaded files
autoscaling:                              # KEDA signal: in-flight requests per replica
  query: 'sum(ovms_current_graphs{namespace="$namespace",model="$model"})'
  scale_target: 4
```

A runtime without `container` runs KServe's default vLLM command. The download job
writes the servable and its `config_all.json` to the model's folder, which KServe mounts
read-only at `/mnt/models`.

### `weight_format` — only for raw Hugging Face repos

`weight_format` is a **conversion** knob for `ovms --pull`, set per category. Use it
(`int8`, `int4`, `fp16`, …) only when `model_id` points at a raw Hugging Face repo that
must be converted and quantized to OpenVINO IR.

> [!IMPORTANT]
> Leave it **unset** for models already published as OpenVINO IR (the `OpenVINO/*`
> repos). Passing `--weight-format` to an already-converted model makes `ovms --pull`
> abort with *"unsupported for OpenVINO models"*.

### API paths

Clients always use the OpenAI `/v1/` paths, which OVMS serves directly. See
[Inference Request Flow](../reference/request_flow.md).

## Which kind of Kubernetes object?

The runtime's `kind` decides what `model-manager` creates:

| Runtime | Object | Template |
|---|---|---|
| vLLM | `LLMInferenceService` | `model_manager/templates/llm-inference-service.yaml` |
| IIMs, OpenVINO | `LLMInferenceService` | the same template, with the runtime's `container` |
| A runtime with `kind: InferenceService` | `InferenceService` | `model_manager/templates/inference-service.yaml`, with the `ClusterServingRuntime` under `model_manager/runtimes/<name>/` |

A model whose runtime changes kind replaces its old object on the next deploy (or undeploy).
