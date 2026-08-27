# Intel® AI for Enterprise Inference

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Component of: Intel® AI for Enterprise Solutions](https://img.shields.io/badge/Component%20of-Intel%C2%AE%20AI%20for%20Enterprise%20Solutions-0068B5)](https://github.com/intel/enterprise-ai-solutions)
[![Platform: Intel Xeon](https://img.shields.io/badge/Platform-Intel%C2%AE%20Xeon%C2%AE-0068B5)](https://www.intel.com/xeon)
[![Serving: KServe](https://img.shields.io/badge/Serving-KServe-326CE5)](https://kserve.github.io/website/)
[![Runtimes: vLLM · OVMS](https://img.shields.io/badge/Runtimes-vLLM%20%C2%B7%20OVMS-purple)](https://vllm.ai)
[![API: OpenAI Compatible](https://img.shields.io/badge/API-OpenAI%20Compatible-green)](https://platform.openai.com/docs/api-reference)

**The inference layer for Intel® AI for Enterprise Solutions. Serve any Hugging Face model on Intel® Xeon® CPUs with one command.**

> Provides the Ansible roles that stand up the model-serving stack — KServe, runtimes, AI gateway, LiteLLM, Langfuse — and **`model-manager`**, a CLI that takes a model from Hugging Face to a live OpenAI-compatible endpoint with NUMA-aware CPU pinning.

> [!IMPORTANT]
> **This repository is not used standalone.** It is a component of the
> [**ai-solutions**](https://github.com/intel/enterprise-ai-solutions)
> platform and is cloned into it at `ext/enterprise.ai-inference/`. Install the core platform
> first — it provisions the Kubernetes cluster and the platform services (cert-manager, Istio,
> MetalLB, Envoy Gateway, PostgreSQL, Keycloak, MinIO, observability) that this layer depends on.
> Every command below runs from the **solutions repo root**, not from here.

---

## What is Intel® AI for Enterprise Inference?

Serving a model on Kubernetes normally means hand-writing manifests, sizing CPU and memory, pinning cores to the right NUMA node, downloading weights into shared storage, and wiring a route through a gateway — for every model.

This repository replaces that with a catalog and a CLI. You describe a model once in `models.yaml`, or point `--id` at a Hugging Face repo, and `model-manager` resolves the configuration, checks the requested CPU fits the node, downloads the weights, creates the serving pod, and registers the model with the gateway.

The Ansible roles here deploy the serving infrastructure itself: KServe for the model CRDs, an in-cluster AI gateway for model-aware routing, and — in `litellm` auth mode — LiteLLM for virtual API keys and Langfuse for request tracing.

The result is a single OpenAI-compatible endpoint. The `model` field in the request body selects which model answers, so any OpenAI SDK client works unchanged.

> Want the full picture? See [Architecture](docs/reference/architecture.md) and [Inference Request Flow](docs/reference/request_flow.md).

## Architecture

An inference request enters through the **LLM gateway**, which authenticates and authorizes it, then routes it to a serving engine — **vLLM** or **OpenVINO™ Model Server** — and returns an OpenAI-compatible response. The gateway also provides model endpoints, user and key management, token telemetry, and monitoring. It all runs on a Kubernetes-orchestrated, Helm-packaged stack over Intel® Xeon® infrastructure.

<p align="center">
  <img src="docs/assets/architecture.png" alt="Intel AI for Enterprise Inference architecture: an inference request and response enter at the top through API applications and services (samples, API apps and functions) over OpenAI-compatible API endpoints; below sits the LLM gateway providing authentication and authorization, model endpoints, user and key management, token telemetry, and monitoring; beneath it the inferencing engines vLLM and OpenVINO Model Server; then the orchestration layer with a Kubernetes orchestrator and Helm charts; and at the base the infrastructure core components — operating system (Ubuntu 22.04/24.04, RHEL) and Xeon software operators and drivers — running on Intel Xeon" />
</p>

> Under the hood there are two gateways: the edge gateway terminates TLS and authenticates, while the in-cluster AI gateway routes to the model the request body names. See the [Architecture deep-dive](docs/reference/architecture.md) and [Inference Request Flow](docs/reference/request_flow.md) for the full component list and execution flow.

---

## Quick Start

**Stands up the inference stack on a freshly prepared machine, then serves an LLM.** In three steps you'll install the inference layer, deploy a model, and see it answer.

> [!NOTE]
> **Prerequisite:** this layer is not standalone. `es_auto_installer.sh` is the installer
> from the [**Intel® AI for Enterprise Solutions**](https://github.com/intel/enterprise-ai-solutions)
> repo, which clones this repository into `ext/enterprise.ai-inference/`. Clone that repo, prep the
> machine, and create an environment, then run every command below from its root:
>
> ```bash
> git clone https://github.com/intel/enterprise-ai-solutions.git
> cd enterprise-ai-solutions
> ./es_auto_installer.sh configure && ./es_auto_installer.sh init local
> ```
>
> This layer needs the cluster, gateway, auth, and storage its dependency layers provide — the
> `install inference` step below auto-pulls them. Full list →
> [solutions Prerequisites](https://github.com/intel/enterprise-ai-solutions/blob/main/docs/quickstart/prerequisites.md).

### Step 1 — Install the inference stack

Deploys KServe, the ServingRuntimes, the AI gateway, and — in `litellm` mode — LiteLLM and Langfuse. No models are deployed yet.

```bash
./es_auto_installer.sh install inference

export KUBECONFIG=$(pwd)/env/local/kubeconfig.yaml
kubectl get pods -n kserve   # controller should be Running
```

Settings for this layer live in `config.yaml` at this repo's root, loaded by the installer as extra vars. See [Configuration](docs/customize/configuration.md) for every option, including the `envoy` / `agentgateway` choice and the cluster CPU-pinning policy.

### Step 2 — Deploy a model

`model-manager` resolves the configuration, verifies the requested CPU fits the node, downloads the weights, creates the serving pod, and registers the model with the gateway.

```bash
./model-manager deploy qwen3-0-6b --wait
```

Or skip the catalog and deploy any Hugging Face model directly:

```bash
./model-manager deploy --id Qwen/Qwen3-0.6B --cpu 8 --memory 16Gi --wait
```

> [!IMPORTANT]
> Gated models (Llama, Mistral, Gemma) require a free [Hugging Face token](https://huggingface.co/settings/tokens).
> Export it as `HF_TOKEN=hf_...` before deploying.

When the command returns it prints the inference endpoint and a ready-to-run `curl` example. Full flag reference → [Deploy a Model](docs/deploy/deploy_models.md).

### Step 3 — Send your first request

With the default `auth_provider: keycloak`, the helper script auto-discovers the gateway IP and fetches a JWT from the cluster:

```bash
source ./ext/enterprise.ai-inference/model_manager/scripts/get-keycloak-token.sh
# Exports: TOKEN, GATEWAY_IP, GATEWAY_DOMAIN
```

Call the model — the `model` field selects which one answers:

```bash
curl -sk --noproxy '*' \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  --resolve "inference.solutions.ai:443:$GATEWAY_IP" \
  -d '{"model":"qwen3-0-6b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}' \
  https://inference.solutions.ai/v1/chat/completions
```

See what is serving:

```bash
kubectl get llminferenceservices,inferenceservices -n llm-inference
```

> [!NOTE]
> **Using `auth_provider: litellm` instead?** LiteLLM becomes the auth boundary and the API is
> served on `litellm.<your-domain>` — not `inference.<your-domain>` — with a LiteLLM master or
> virtual key instead of a JWT. Full walkthrough →
> [Accessing Models](docs/deploy/accessing_models.md).

Everything speaks the OpenAI-compatible API, so any tool that works with OpenAI works against your stack.

---

## Advanced

The Quick Start deploys one catalog model with defaults. From here you can change the engine and its version, add models, tune CPU and memory, switch the AI gateway or the CPU-pinning policy, and control what deploys automatically at install time.

| Goal | Guide |
|---|---|
| Deploy, update, and remove models | [Deploy a Model](docs/deploy/deploy_models.md) |
| Endpoints and authentication modes | [Accessing Models](docs/deploy/accessing_models.md) |
| Add models — the `models.yaml` catalog | [Model Catalog](docs/customize/catalog.md) |
| Engines, versions, and per-workload defaults | [Runtimes](docs/customize/runtimes.md) |
| All configuration options | [Configuration](docs/customize/configuration.md) |
| Components, roles, and how this repo plugs in | [Architecture](docs/reference/architecture.md) |
| How a request reaches a model | [Inference Request Flow](docs/reference/request_flow.md) |
| NUMA-aware CPU pinning | [CPU Pinning & NUMA](docs/reference/cpu_pinning.md) |
| Something isn't working | [Troubleshooting](docs/troubleshooting.md) |

---

## License

Licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Links

- [Documentation Index](docs/README.md)
- [GitHub Repository](https://github.com/intel/enterprise-inference)
- [Intel® AI for Enterprise Solutions (core platform)](https://github.com/intel/enterprise-ai-solutions)
- [Architecture](docs/reference/architecture.md)
- [Inference Request Flow](docs/reference/request_flow.md)

---

*Intel® and Intel® Xeon® are registered trademarks of Intel Corporation or its subsidiaries. Licensed under the Apache License, Version 2.0.*
