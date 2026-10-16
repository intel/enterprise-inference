# Deploy a Model

`model-manager` is a CLI that takes a model from Hugging Face to a live
OpenAI-compatible endpoint. It exposes two lifecycle commands — **`deploy`** and
**`undeploy`** — and delegates status, scaling, and logs to `kubectl`, since the
serving objects are standard Kubernetes CRDs.

> [!IMPORTANT]
> Run it from the **solutions repo root** as `./model-manager` (a thin wrapper pointing at
> `ext/enterprise.ai-inference/model_manager/`). It operates against a **running cluster**
> provisioned by the core platform install — see [Quick Start](../../README.md#quick-start).

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
./model-manager deploy <name>            # deploy a model from the catalog
./model-manager deploy --id <hf/repo>    # deploy any Hugging Face model (no catalog entry)
./model-manager undeploy <name>          # stop serving a model (weights stay on disk)
./model-manager undeploy all             # stop serving every model
```

Deploy is safe to re-run. Running it again on a model that is already up **updates it
in place** — change `--cpu` or `--replicas` and re-deploy. It won't error, and it
won't re-download weights already on disk.

If nothing about the rendered spec changed, the re-deploy is a **no-op**:
model-manager diffs the spec server-side and skips both the pre-flight check and the
apply. Set `MM_FORCE_REDEPLOY=1` to force a full re-apply regardless.

## Deploy options

| Flag | What it does |
|------|--------------|
| `--id <hf/repo>` | Deploy any Hugging Face model directly, without a catalog entry |
| `--name <name>` | Set the name to serve under (use with `--id`) |
| `--as <name>` | Deploy a catalog model under a different name — run several variants at once |
| `--server <name>` | Which engine serves the model: `vllm` (default) or `openvino` (alias: `--runtime`) |
| `--server-version <v>` | Which version of that engine to use — see [below](#pick-an-engine-and-version) |
| `--cpu N` | Number of CPU cores to give the model |
| `--memory NGi` | Memory limit, e.g. `32Gi` |
| `--replicas N` | How many copies to run (default: 1) |
| `--tp N` | Split one model across N CPU sockets for more throughput (advanced) |
| `--category <cat>` | Force the workload type: `llm`, `embed`, `rerank`, or `vlm` |
| `--node <name>` | Pin the model to a specific cluster node |
| `--cpu-policy <p>` | Override the CPU-pinning policy for this model: `nri-balloons`, `kubelet-static`, or `best-effort`. Must match how the cluster was installed — see [CPU Pinning & NUMA](../reference/cpu_pinning.md) |
| `--routing <mode>` | How requests reach the model: `epp` or `direct` (advanced) |
| `--env KEY=VAL` | Set an environment variable on the pod (repeatable) |
| `--arg "--flag=val"` | Pass an extra engine flag, e.g. a vLLM option (repeatable) |
| `--wait` / `--wait-timeout N` | Wait until the model is ready before returning (default: 900s) |
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

## Ad-hoc deploy from Hugging Face

You don't need a catalog entry to try a model. Point `--id` at any Hugging Face repo
and model-manager infers the workload type (chat model, embedding, or reranker) from
the model's metadata:

```bash
./model-manager deploy --id Qwen/Qwen3-0.6B --cpu 8 --memory 16Gi --wait
./model-manager deploy --id meta-llama/Llama-3.2-3B-Instruct --cpu 16 --memory 32Gi --wait
./model-manager deploy --id BAAI/bge-base-en-v1.5 --cpu 4 --memory 8Gi --wait
```

## Gated models (Hugging Face token)

Gated models (Llama, Mistral, Gemma) require a [Hugging Face token](https://huggingface.co/settings/tokens):

```bash
export HF_TOKEN="hf_your_token_here"
./model-manager deploy --id meta-llama/Llama-3.2-3B-Instruct --cpu 16 --memory 32Gi --wait
```

The download Job receives the token automatically. Public models work without it. A
gated model **without** a token fails fast with a clear "model is gated — set
HF_TOKEN" message, and during auto-deploy is reported without aborting the install.

## Models with custom code (`--trust-remote-code`)

Some Hugging Face models ship custom modeling code that vLLM only loads with
`--trust-remote-code`. Pass it like any other engine flag — via `--arg`, or the
catalog's per-model `args:` — there is no dedicated option:

```bash
./model-manager deploy --id <org/model> --arg "--trust-remote-code" --wait
```

When the flag is present from any layer (category, version, per-model `args:`, or
`--arg`), model-manager arranges offline-safe serving automatically:

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

# Change the number of replicas
kubectl patch llminferenceservice <name> -n llm-inference \
  --type merge -p '{"spec":{"replicas":2}}'

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
3. **Ensure storage** — make sure the shared `model-store` PVC is ready.
4. **Download weights** — a one-off Kubernetes Job pulls the model from Hugging Face
   onto that volume, skipped if the weights are already there.
5. **Deploy** — create the serving pod: `LLMInferenceService` for vLLM,
   `InferenceService` for OpenVINO.
6. **Register** — wire the model into the AI gateway (or LiteLLM) so it can receive
   requests.

Next: [Accessing Models](accessing_models.md) for endpoints and credentials.
