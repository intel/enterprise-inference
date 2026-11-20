# Deploy a Model

`model-manager` is a CLI that takes a model from Hugging Face to a live
OpenAI-compatible endpoint. It exposes two lifecycle commands — **`deploy`** and
**`undeploy`** — and delegates status, scaling, and logs to `kubectl`, since the
serving objects are standard Kubernetes CRDs.

> [!IMPORTANT]
> Run it from the **Toolkit Manager repo root** as `./model-manager` (a thin wrapper pointing at
> `ext/enterprise.ai-inference/model_manager/`). It operates against a **running cluster**
> Toolkit Manager installed — see [Quick Start](../../README.md#quick-start).

It needs `kubectl`, `curl`, and **Python 3.9 or newer with PyYAML**. The installer's `.venv`
is picked up automatically (as is any active virtualenv); otherwise install the dependencies
for the system `python3`:

```bash
pip install -r ext/enterprise.ai-inference/model_manager/requirements.txt
```

## deploy / undeploy

**Deploy** downloads a model's weights, starts a serving pod, and registers it with
the gateway so it can take requests. **Undeploy** removes the pod but leaves the
downloaded weights on disk, so re-deploying later is fast.

```bash
./model-manager deploy <name>              # deploy a model from the catalog
./model-manager deploy --hfid <hf/repo>    # deploy any Hugging Face model (no catalog entry)
./model-manager undeploy <name>            # stop serving a model (weights stay on disk)
./model-manager undeploy all               # stop serving every model
./model-manager list                       # catalog models and their status, plus models deployed outside it
```

`list --catalog` shows only `models.yaml` (no cluster needed); `list --deployed` only
what is running.

Deploy is safe to re-run. Running it again on a model that is already up **updates it
in place** — change `--cpu` or `--replicas` and re-deploy. It won't error, and it
won't re-download weights already on disk.

If nothing about the rendered spec changed, the re-deploy is a **no-op**:
model-manager diffs the spec server-side and skips both the pre-flight check and the
apply. Set `MM_FORCE_REDEPLOY=1` to force a full re-apply regardless.

## Deploy options

| Flag | What it does |
|------|--------------|
| `--hfid <hf/repo>` | Deploy any Hugging Face model directly, without a catalog entry (`--id` still works but is deprecated) |
| `--name <name>` | Set the name to serve under (use with `--hfid`) |
| `--as <name>` | Deploy a catalog model under a different name — run several variants at once |
| `--server <name>` | Which engine serves the model: `vllm` (default), `openvino`, `vllm-cuda` or `sglang-xpu` (alias: `--runtime`) |
| `--server-version <v>` | Which version of that engine to use — see [below](#pick-an-engine-and-version) |
| `--cpu N` | Number of CPU cores to give the model. `--cpu auto` sizes it to a NUMA node — see [CPU auto](../customize/catalog.md#cpu-auto) |
| `--memory NGi` | Memory limit, e.g. `32Gi` |
| `--replicas N` | How many copies to run (default: 1); a fixed count, which turns autoscaling off |
| `--min-replicas N` / `--max-replicas N` | Autoscale between N copies (default min: 1); a max above the min turns it on. Needs `keda_enabled: true` and KServe 0.21 (`envoy_gateway_upgrade: true`); not on OpenShift yet |
| `--scale-target N` | Load per copy that triggers a scale-up: waiting requests for vLLM (default: 2), in-flight requests for OpenVINO (default: 4) |
| `--tp N` | Split one model across N CPU sockets for more throughput (advanced); on a GPU engine, across N GPUs, which also sets `--gpus` |
| `--gpus N` | GPUs per pod for a GPU engine (`vllm-cuda`, `sglang-xpu`); also the default `--tp` — see [below](#deploy-on-gpu-nodes) |
| `--category <cat>` | Force the workload type: `llm`, `embed`, `rerank`, or `vlm` |
| `--node <name>` | Pin the model to a specific cluster node |
| `--cpu-policy <p>` | Override the CPU-pinning policy for this model: `nri-balloons`, `kubelet-static`, or `best-effort`. Must match how the cluster was installed — see [CPU Pinning & NUMA](../reference/cpu_pinning.md) |
| `--routing <mode>` | How requests reach the model: `epp` or `direct` (advanced) |
| `--env KEY=VAL` | Set an environment variable on the pod (repeatable) |
| `--arg "--flag=val"` | Pass an extra engine flag, e.g. a vLLM option (repeatable) |
| `--wait` / `--wait-timeout N` | Wait until the model is ready before returning (default: 900s; a runtime can set its own, e.g. 3600s for `sglang-xpu`) |
| `--dry-run` | Render and show what would be deployed, without changing anything |

Run `./model-manager --help` for the authoritative list.

## Pick an engine and version

A model can run on different engines (**servers**) and on different **versions** of an
engine. This lets you upgrade deliberately, or A/B two versions side by side.

```bash
# Use the model's default server and version (from the catalog)
./model-manager deploy qwen3-0-6b --wait

# Choose a specific engine version
./model-manager deploy qwen3-0-6b --server-version 0.19.1 --wait

# Run two versions at the same time, under different names
./model-manager deploy qwen3-0-6b --server-version 0.24.0 --as qwen3-v24 --wait
./model-manager deploy qwen3-0-6b --server-version 0.19.1 --as qwen3-v19 --wait
```

The catalog decides which versions a model is *allowed* to use — asking for one that
isn't listed fails with a clear message rather than deploying something untested. See
[Model Catalog](../customize/catalog.md#choose-which-server-and-version-a-model-may-use).

## Deploy on GPU nodes

The catalog ships GPU engines, `vllm-cuda` (NVIDIA) and `sglang-xpu` (Intel Arc Pro B-Series, see
[SGLang on Intel Arc Pro B-Series](../customize/runtimes.md#sglang-on-intel-arc-pro-b-series-sglang-xpu)),
alongside the default CPU `vllm`.
A model runs on GPU by selecting that runtime and, optionally, a GPU count:

```yaml
- name: llama-3-1-70b-instruct-cuda
  model_id: meta-llama/Llama-3.1-70B-Instruct
  category: llm
  runtime: vllm-cuda
  gpus: 8          # per pod; also the default --tensor-parallel-size
  memory: 128Gi    # host RAM for weight loading — not the accelerator
```

What `vllm-cuda` changes in the rendered manifest, relative to CPU `vllm`:

- `resources.requests` and `.limits` gain `nvidia.com/gpu: "<gpus>"` — the
  scheduler only places the pod on a node that advertises that resource.
- A `nvidia.com/gpu` **toleration** (`NoSchedule`), so tainted GPU nodes accept it.
- A memory-backed **`/dev/shm`** volume (`shm_size`, default 10Gi) — required
  by NCCL for tensor parallelism.
- `--tensor-parallel-size <gpus>` is injected when `gpus > 1` and no TP flag
  was given.
- **No CPU limit**, so the pod is Burstable and kubelet's static CPU manager
  does not pin it.
- **No CPU pinning.** NRI balloon / kubelet-static annotations, their
  pre-flight checks, and the AMX node requirement are all skipped: they are
  Xeon placement controls and do not apply to a GPU pod. A GPU pre-flight
  runs instead (see [The GPU pre-flight](#the-gpu-pre-flight)), and the CPU and
  memory check counts only the GPU nodes.

### The GPU pre-flight

A GPU model deploys on any cluster whose nodes advertise its runtime's
`accelerator_resource`, rack-scale or not; there is no switch to turn GPU
deploys on. Before downloading weights, each new replica needs a Ready node, not
cordoned or tainted against the pod, with at least `gpus` free `nvidia.com/gpu`,
counting what running pods already hold (a pod's GPUs cannot span nodes).
Otherwise the deploy stops: on a CPU-only cluster, or when a GPU node has not
joined the cluster or its device plugin is not ready yet. It also stops when
the nodes and pods cannot be listed. With `--node`, which pins the pod, only
that node counts. `MM_PREFLIGHT=warn` proceeds anyway; `MM_SKIP_PREFLIGHT=1`
skips the check.

A re-deploy that changes more than `--replicas` may restart the pod: the
rollout starts the new one next to the running one, so its GPUs must be free
too. When the model already fills its node, run `./model-manager undeploy <name>`
first, or set `MM_PREFLIGHT=warn` to apply anyway (for example for a label-only
change; a new pod then waits for GPUs). A change of `--replicas` alone keeps the
running pods: a scale-down needs no free GPUs, and a scale-up needs them only
for the added replicas. The pre-flight checks room for the new replicas next to the
running ones (at least one new pod when the change restarts them), and that every replica
fits once the running pods are gone, but not every step of the rolling update in between.
Three changes can still stall the rollout with a new pod Pending: raising `--gpus` or
moving the model with `--node` when it has two or more replicas, and raising `--replicas`
together with a change that restarts the pods. Undeploy the model first for those, or for
the last one change `--replicas` in a deploy of its own. A stalled rollout stays as it is:
deploy the model again with its previous settings and `MM_PREFLIGHT=warn` (the pre-flight
counts the stalled pods as running), or undeploy it and deploy it again.
GPU models take a fixed `--replicas`: autoscaling
(`--min-replicas`/`--max-replicas`) is not available for GPU runtimes (`vllm-cuda`, `sglang-xpu`) yet.

The model store must be mountable on a GPU node. On node-local storage
(`local-path`), a store already bound to a node without the GPUs stops the
deploy: GPU models need shared (ReadWriteMany) storage, or the store on a GPU
node. A store that binds on first use (and is not ReadWriteMany) is created
where the download Job runs, so the pre-flight pins the Job to the GPU nodes
that fit (not when it is skipped, or when `MM_PREFLIGHT=warn` proceeds with no
GPU node that fits), and warns: on node-local storage every model,
CPU ones included, must then run on that node.
GPU replicas on more than one node need a ReadWriteMany store: `storage_backend`
`nfs`, `ceph` (CephFS) or `netapp-trident` (NAS drivers), or `custom` unless
`storage_custom_rwx: false`. A ReadWriteOnce block volume attaches to one node
at a time, and the pre-flight does not check where it is attached.

`--tp N` without `--gpus` also sets the GPU count to N. A tensor parallel size
larger than the GPU count is refused; a smaller one warns about the unused GPUs.
The count covers tensor parallelism only. With pipeline or data parallelism passed
through `--arg`, set `--gpus` to the total GPUs the pod needs, and the tensor
parallel size with `--tp` or a `--tensor-parallel-size` in `--arg` (the long
form). Otherwise it is the catalog's value, or the GPU count. The "GPUs unused"
warning then counts only tensor parallelism, so it can be ignored.

```bash
./model-manager deploy llama-3-1-70b-instruct-cuda --wait
./model-manager deploy qwen3-0-6b-cuda --dry-run        # preview the manifest
./model-manager deploy qwen3-0-6b-cuda --gpus 2         # override the catalog count
```

### GPU models in the catalog

None deploys at install time (`autodeploy: false`). Like CPU models, they are
routed through the AI Gateway, and registered with LiteLLM when `auth_type` is
`litellm_vkey`.

| Name | Model | GPUs | Notes |
|---|---|---|---|
| `qwen3-0-6b-cuda` | Qwen/Qwen3-0.6B | 1 | Smoke test |
| `llama-3-1-70b-instruct-cuda` | meta-llama/Llama-3.1-70B-Instruct | 8 | Gated: needs `HF_TOKEN` |
| `nemotron-3-5-lightning-30b-a3b-cuda` | nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | 1 | |
| `nemotron-3-super-120b-a12b-cuda` | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 | 4 | Hopper or Blackwell. On B200/B300 the model card fits it on one GPU: `--gpus 1` |
| `nemotron-3-ultra-550b-a55b-cuda` | nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 | 8 | NVFP4: Blackwell only (B200, B300) |

The Nemotron models are reasoning models (`--reasoning-parser nemotron_v3`) with
tool calling (`--tool-call-parser qwen3_coder`), using the flags from their model
cards.

### Model store size

All downloaded weights share one PVC, `model-store`, sized by
`llm_services_model_store_size` (default `200Gi`, see
[Model-store size](../customize/catalog.md#model-store-size)). That holds
Llama 3.1 70B (~141 GB) or Nemotron 3 Super FP8 (~128 GB), but not both, and
not Nemotron 3 Ultra NVFP4 (~352 GB). For all of them, set
`llm_services_model_store_size: 1Ti` in `config.inference.yaml` before the
first inference install. A model that does not fit is refused before its
download starts. A download that outlasts the deploy's two-hour wait resumes
when the deploy is run again.

Pod security is unchanged: the upstream `vllm/vllm-openai` image runs under the
same `runAsNonRoot` / `drop: ALL` context as the CPU image, with its caches
redirected to `/tmp` via the runtime's env (`HOME`, `XDG_CACHE_HOME`,
`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`).

## Ad-hoc deploy from Hugging Face

You don't need a catalog entry to try a model. Point `--hfid` at any Hugging Face repo
and model-manager infers the workload type (chat model, embedding, or reranker) from
the model's metadata:

```bash
./model-manager deploy --hfid Qwen/Qwen3-0.6B --cpu 8 --memory 16Gi --wait
./model-manager deploy --hfid meta-llama/Llama-3.2-3B-Instruct --cpu 16 --memory 32Gi --wait
./model-manager deploy --hfid BAAI/bge-base-en-v1.5 --cpu 4 --memory 8Gi --wait
```

## Gated models (Hugging Face token)

Gated models (Llama, Mistral, Gemma) require a [Hugging Face token](https://huggingface.co/settings/tokens):

```bash
export HF_TOKEN="hf_your_token_here"
./model-manager deploy --hfid meta-llama/Llama-3.2-3B-Instruct --cpu 16 --memory 32Gi --wait
```

The download Job receives the token automatically. Public models work without it. A
gated model **without** a token fails fast with a clear "model is gated — set
HF_TOKEN" message, and during auto-deploy is reported without aborting the install.

## Models with custom code (`--trust-remote-code`)

Some Hugging Face models ship custom modeling code that vLLM only loads with
`--trust-remote-code`. Pass it like any other engine flag — via `--arg`, or the
catalog's per-model `args:` — there is no dedicated option:

```bash
./model-manager deploy --hfid <org/model> --arg "--trust-remote-code" --wait
```

When the flag is present from the category, the version, per-model `args:`, or
`--arg`, model-manager arranges offline-safe serving automatically:

- the **download Job** prefetches the `*.py` from every repo referenced in the model's
  `config.json` `auto_map`, then rewrites `auto_map` to point at the local copies, and
- the **serving pod** gets `HF_HUB_OFFLINE=1` so vLLM loads that prefetched code
  instead of reaching the Hub at runtime (override with `--env HF_HUB_OFFLINE=0`).

## Status, scaling, and logs

model-manager handles deploy and undeploy only. For everything else use `kubectl` —
the models are standard Kubernetes objects:

```bash
# See what's currently serving
kubectl get llminferenceservices,inferenceservices -n llm-inference

# Change the number of replicas of a fixed-count model (an autoscaled one rejects
# spec.replicas: redeploy it with --replicas or --max-replicas instead)
kubectl patch llminferenceservice <name> -n llm-inference \
  --type merge -p '{"spec":{"replicas":2}}'

# Autoscaled models: their KEDA ScaledObject and HPA
kubectl get scaledobject,hpa -n llm-inference

# Follow a model's logs
kubectl logs -n llm-inference -l app.kubernetes.io/name=<name> -f

# List the model names in the catalog
yq '.models[].name' env/<env>/models.yaml
```

## How a deploy works

Each `deploy` runs these steps in order:

1. **Resolve config** — combine the catalog defaults, the model's entry, and any CLI
   flags into one final configuration.
2. **Capacity check** — confirm the requested CPU fits the node's physical cores
   before changing anything. See [CPU Pinning & NUMA](../reference/cpu_pinning.md).
   A GPU model checks free GPUs and the model store instead (see [The GPU pre-flight](#the-gpu-pre-flight)).
3. **Ensure storage** — make sure the shared `model-store` PVC is ready.
4. **Download weights** — a one-off Kubernetes Job pulls the model from Hugging Face
   onto that volume, skipped if the weights are already there.
5. **Deploy** — create the serving pod as an `LLMInferenceService`, for vLLM and OpenVINO
   alike; KServe's scheduler then routes each request to a replica.
6. **Register** — wire the model into the AI gateway (or LiteLLM) so it can receive
   requests.

Next: [Accessing Models](accessing_models.md) for endpoints and credentials.
