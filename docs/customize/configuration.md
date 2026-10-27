# Configuration

## What is this?

`config.yaml` at this repo's root holds the settings for the **inference layer** —
which AI gateway to use, which KServe and LiteLLM versions to install, hostnames, and
the cluster's CPU-pinning policy.

The core installer loads it as Ansible extra vars:

```
-e @ext/enterprise.ai-inference/config.yaml
```

### How it fits together

Three layers, each overriding the one before it:

```
roles/<name>/defaults/main.yaml     role defaults — the baseline
        ↓
config.yaml  (this repo)            inference-layer settings
        ↓
env/<name>/global_config.yaml       platform settings (base_domain_name, auth_provider, …)
        ↓
-e on the command line              one-off override
```

> [!IMPORTANT]
> `config.yaml` is checked into this repo, so it applies to **every** environment.
> Anything environment-specific — domain, auth provider, storage — belongs in the
> platform's `env/<name>/global_config.yaml` instead. See the
> [solutions Configuration Reference](https://github.com/intel/enterprise-ai-solutions/blob/main/docs/customize/configuration.md).

### Do I need to change anything?

No. The defaults deploy a working inference stack. Reach for this file when:

| If you want to… | Change |
|---|---|
| Use agentgateway instead of Envoy AI Gateway | `ai_gateway_provider: "agentgateway"` |
| Turn the content-safety guardrail on (off by default; the Agent Toolkit turns it on) | `guardrails_enabled: true` |
| Route by prompt complexity inside LiteLLM | `hybrid_router_enabled: true` |
| Route with the vLLM Semantic Router | `vllm_sr_enabled: true` |
| Apply changed LiteLLM settings to an installed release | `litellm_reconcile: true` |
| Auto-deploy catalog models during install | `llm_services_deploy_models: true` |
| Make an auto-deploy failure abort the install | `model_deploy_strict: true` |
| Pin a component version | `kserve_version`, `litellm_version`, `envoy_ai_gateway_version` |
| Change a service hostname | `inference_hostname`, `litellm_hostname`, `langfuse_hostname` |
| Change CPU pinning behaviour | `kubernetes_cpu_policy` |
| Serve models from a different namespace | `llm_services_namespace` |

## AI gateway

Which in-cluster AI gateway sits behind the edge gateway and dispatches to the KServe
model backends. Both speak the OpenAI protocol on the same unified URL and route to
`InferencePool`/`Service` backends — only the data plane and CRDs differ.

```yaml
ai_gateway_provider: "envoy"          # envoy | agentgateway
```

| Value | What it deploys |
|---|---|
| `envoy` (default) | Envoy AI Gateway — `AIGatewayRoute` CRDs, EPP endpoint picker |
| `agentgateway` | agentgateway — native Gateway API `HTTPRoute` → `InferencePool`, plus MCP / A2A support for the agentic layer |

```yaml
# Envoy AI Gateway (when ai_gateway_provider=envoy)
envoy_ai_gateway_version: "0.6.0"
envoy_ai_gateway_namespace: "envoy-ai-gateway-system"

# agentgateway (when ai_gateway_provider=agentgateway)
# agentgateway_version: "1.3.1"
# agentgateway_namespace: "agentgateway-system"
```

## Hostnames

Every hostname defaults to a prefix on the platform's `base_domain_name`.

| Setting | Default | Serves |
|---|---|---|
| `inference_hostname` | `inference.<base_domain_name>` | Model API in `keycloak` mode |
| `inference_prefix` | `inference` | Prefix used when `inference_hostname` is unset |
| `litellm_hostname` | `litellm.<base_domain_name>` | Model API in `litellm` mode |
| `langfuse_hostname` | `langfuse.<base_domain_name>` | Langfuse UI |
| `vllm_sr_dashboard_hostname` | `vllm-sr.<base_domain_name>` | vLLM Semantic Router dashboard (when enabled) |

> [!NOTE]
> The two auth modes serve on **different** hostnames. See
> [Accessing Models](../deploy/accessing_models.md).

## Keycloak

The realm and client are owned by this solution; Keycloak itself is deployed by the
platform.

```yaml
keycloak_realm: "inference"
keycloak_client_id: "inference-client"
# keycloak_token_lifespan: 900   # JWT access token lifespan in seconds (default: 15 min)
```

These roles run only when `auth_provider` is not `litellm`.

## KServe

```yaml
# kserve_version: "0.15.1"
# kserve_deploy_mode: "Standard"     # Standard | Knative
# kserve_namespace: "kserve"
```

KServe runs in **Standard** mode — direct pod-based serving. It requires cert-manager and
Envoy Gateway, both installed by the core platform. Standard is the validated and
supported mode.

## LiteLLM and Langfuse

Deployed only when `auth_provider: litellm`.

```yaml
# litellm_version: "v1.83.14-stable.patch.2"
# litellm_namespace: "litellm"
# litellm_cache_enabled: true
# litellm_cache_ttl: 600

# langfuse_namespace: "langfuse"
# langfuse_admin_email: "admin@<base_domain_name>"
# langfuse_admin_password: set via the LANGFUSE_ADMIN_PASSWORD env var
```

The LiteLLM master key is generated at install and stored in the `litellm-master-key`
secret in the `litellm` namespace. Langfuse's admin password comes from the
`LANGFUSE_ADMIN_PASSWORD` environment variable, not from this file.

An installed LiteLLM release is left as it is on later installs, except that one whose
config lacks the guardrail's guard host allowlist is upgraded. `litellm_reconcile: true`
upgrades it with the current settings — needed after changing the hybrid router or the
sizing on a running stack.

```yaml
# litellm_reconcile: true       # upgrade an existing release with the current settings
# guardrails_enabled: true      # content-safety guardrail (default: off)
# guardrail_model: "qwen3guard" # guard model from the catalog
```

The guardrail screens every LiteLLM request with a guard model that llm_services deploys
alongside the other models → [Guardrails](guardrails.md).

## Optional routing

Both are off by default and leave direct model calls unchanged.

```yaml
# Hybrid router: one virtual model in LiteLLM picks a model per prompt complexity tier.
# hybrid_router_enabled: true
# hybrid_router_tier_candidates:
#   SIMPLE:  [qwen3-4b]
#   COMPLEX: [qwen3-coder-30b]

# vLLM Semantic Router (ai_gateway_provider: envoy only)
# vllm_sr_enabled: true
# vllm_sr_route_models: [qwen3-4b, qwen3-coder-30b]
# vllm_sr_litellm_model_name: "semantic-router"   # with auth_provider=litellm
# vllm_sr_dashboard_enabled: true                 # vllm-sr.<base_domain_name>
```

Details → [Hybrid Router](hybrid-router.md), [vLLM Semantic Router](vllm-semantic-router.md).

## CPU policy

A single switch controlling all CPU-pinning infrastructure. Dependent settings — NRI
install, kubelet flags, model annotations — are derived from it.

```yaml
kubernetes_cpu_policy: "nri-balloons"   # nri-balloons | kubelet-static | best-effort
```

| Value | Behaviour |
|---|---|
| `nri-balloons` (default) | NRI balloons resource policy — NUMA-aware pinning |
| `kubelet-static` | kubelet static CPU manager — Guaranteed QoS pinning |
| `best-effort` | No pinning, standard scheduler |

> [!WARNING]
> This must match the value the cluster was **installed** with. It decides both how
> kubelet is configured (solutions repo) and how model manifests are rendered (this
> repo). Changing it on an existing cluster requires re-running
> `./es_auto_installer.sh install kubernetes` so kubelet is reconfigured.

Tuning, per policy:

```yaml
# nri-balloons only
# nri_reserved_cpu_list: ""            # Linux cpuset for system CPUs (auto-sized if empty)
# nri_deployment_profile: "standard"   # minimal(8) | standard(12) | observability(16) | full(20)

# kubelet-static only
# mm_kubelet_threads_per_core: 1       # set to 1 on an SMT-disabled cluster
```

Under `kubelet-static`, model pods are rendered as Guaranteed QoS (integer `cpu`,
requests == limits) so kubelet's static CPU manager grants them exclusive whole cores.
With `full-pcpus-only` enabled (the solutions default) a pod's `cpu` request must be a
multiple of the SMT width or kubelet rejects it with `SMTAlignmentError`; model-manager
rounds `cpu` up to satisfy this.

Full detail → [CPU Pinning & NUMA](../reference/cpu_pinning.md).

## LLM services and model deployment

```yaml
# llm_services_namespace: "llm-inference"
# llm_services_deploy_runtimes: true   # deploy namespace-scoped ServingRuntimes

llm_services_deploy_models: false      # run the model auto-deploy phase
# llm_services_autodeploy_models: []   # catalog models deployed whatever the line above says
# model_deploy_strict: false           # true = an auto-deploy failure aborts the install
# model_deploy_flags: ""               # extra flags for every auto-deploy invocation
```

> [!IMPORTANT]
> `llm_services_deploy_models` ships as **`false`**, overriding the role default of
> `true`. It gates the whole auto-deploy phase, so flagging a model `autodeploy: true`
> has no effect until you also set this to `true`. See
> [Auto-deploy on install](catalog.md#auto-deploy-on-install).

`llm_services_autodeploy_models` is for a solution layer that needs a model to be up — the
agent toolkit sets its default model here. Those models, and the guardrail's guard model
when the guardrail is on (deployed first), deploy even with `llm_services_deploy_models:
false`. A name `env/<env>/models.yaml` lacks is taken from the shipped
`model_manager/models.yaml`; a name neither has is reported — fatal for the guard model, or
with `model_deploy_strict: true`.

`llm_services_deploy_runtimes` deploys `files/runtimes/*.yaml` as namespace-scoped
`ServingRuntime`s. These are needed for `InferenceService` CRs (OpenVINO);
`LLMInferenceService` CRs (vLLM) ignore them.

`model_deploy_flags` is appended to each `model-manager deploy <name> --wait` invocation
during auto-deploy — e.g. `"--wait-timeout 1800"` to allow longer for large models, or
`"--dry-run"` to render manifests without applying.

## Node Feature Discovery

```yaml
# nfd_enabled: true
# nfd_namespace: "node-feature-discovery"
# nfd_version: "0.17.1"
```

NFD runs as a DaemonSet and labels every node with its hardware capabilities:

```
feature.node.kubernetes.io/cpu-security.amx.enabled=true
feature.node.kubernetes.io/cpu-cpuid.AVX512F=true
feature.node.kubernetes.io/memory-numa.node_count=2
```

model-manager consumes these in `nodeAffinity` rules — notably the catalog's
`require_amx: true` default, which keeps model pods off nodes without Intel® AMX.
Disabling NFD means those affinity rules cannot be satisfied.

## Environment variables

`model-manager` reads these at runtime. The commonly useful ones:

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | Hugging Face token for gated models |
| `MM_CONFIG` | Path to the model catalog (`models.yaml`) |
| `MM_CHAT_TEMPLATES` | Directory a model's relative `chat_template_file` is read from (default `model_manager/chat-templates`) |
| `MM_GLOBAL_CONFIG` | Path to the platform's `global_config.yaml` |
| `MM_INFERENCE_HOSTNAME` | Override the resolved endpoint hostname in either auth mode |
| `MM_INFERENCE_GATEWAY` / `MM_INFERENCE_GATEWAY_NS` | Gateway a model's routes attach to (default `ai-gateway` in `llm-inference`) |
| `MM_INFERENCE_GATEWAY_SVC_NS` | Namespace of that gateway's Service, which LiteLLM calls (default: `MM_INFERENCE_GATEWAY_NS`, else `envoy-gateway-system`) |
| `MM_CPU_POLICY` | Override `kubernetes_cpu_policy` for one invocation |
| `MM_NRI_ENABLED` | Set `false` to skip NRI CPU-pinning annotations |
| `MM_NRI_PREFLIGHT` | `warn` — report a capacity shortfall but proceed |
| `MM_NRI_SKIP_PREFLIGHT` | `1` — skip the capacity check entirely |
| `MM_FORCE_REDEPLOY` | `1` — re-apply a model even if its rendered spec is unchanged |
| `MM_HTTP_PROXY` / `MM_HTTPS_PROXY` / `MM_NO_PROXY` | Proxy for reaching Hugging Face (Hub checks, the download job, OVMS pods). `models.yaml` `network:` takes precedence; `HTTP_PROXY` / `http_proxy` etc. are the fallback |
| `SKIP_MODEL_DEPLOYMENT` | `true` — skip the auto-deploy phase for one install run |

Run `./model-manager --help` for the full surface.
