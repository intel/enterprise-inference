# Architecture

This repository owns everything related to model inference — the Kubernetes serving
stack and the CLI that deploys and manages models at runtime.

## Components

| Component | Path | Purpose |
|-----------|------|---------|
| **KServe** | `roles/kserve/` | Model-serving platform — KServe controller, `LLMInferenceService` CRD, LWS operator for distributed inference |
| **Envoy AI Gateway** | `roles/envoy_ai_gateway/` | AI-aware routing — token-based load balancing, model-level rate limiting, unified OpenAI endpoint via `AIGatewayRoute` |
| **agentgateway** | `roles/agentgateway/` | Alternative AI gateway — native Gateway API `HTTPRoute` → `InferencePool`, plus MCP / A2A support |
| **LiteLLM** | `roles/litellm/`, `charts/litellm-helm/` | OpenAI-compatible proxy — virtual API keys, per-key budgets and rate limits |
| **Langfuse** | `roles/langfuse/` | LLM observability — request traces, token usage, prompt management |
| **LLM services** | `roles/llm_services/` | Provisions the inference namespace and ServingRuntimes, and runs the model auto-deploy phase |
| **Keycloak config** | `roles/keycloak_config/` | Inference realm + client for sign-in and JWT-based API auth (both auth modes) |
| **NFD** | `roles/nfd/` | Node Feature Discovery — labels nodes with hardware capabilities (AMX, AVX512, NUMA count) |
| **vLLM Semantic Router** | `roles/vllm_sr/`, `charts/vllm-sr/` | Opt-in classifier-based routing on a dedicated Envoy AI Gateway, plus its dashboard |
| **Model Manager** | `model_manager/` | Bash CLI (`model-manager`) + runtime and manifest templates for the model lifecycle |

## Ansible roles

Each role is a component in the `inference` layer, registered in `components.yaml` and
resolved by the core installer in dependency order.

| Role | Enabled when | Depends on |
|------|--------------|------------|
| `nfd` | `nfd_enabled` (default `true`) | — |
| `keycloak_config` | `keycloak_enabled` (default `true`; both auth modes) | — |
| `envoy_ai_gateway` | `kserve_enabled` and `ai_gateway_provider == envoy` | — |
| `agentgateway` | `ai_gateway_provider == agentgateway` | — |
| `kserve` | `kserve_enabled` (default `true`) | `envoy_ai_gateway`, `agentgateway` |
| `litellm` | `auth_type == litellm_vkey` or `litellm_enabled` | — |
| `langfuse` | `litellm` is enabled and `langfuse_enabled` (default `true`) | `litellm` |
| `llm_services` | `kserve_enabled` (default `true`) | `kserve` |
| `vllm_sr` | `vllm_sr_enabled` (default `false`) and `ai_gateway_provider == envoy` | `llm_services` |

> [!NOTE]
> `nri_cpu_balloons` is **not** here — it lives in the solutions repo, because it
> configures kubelet-level settings on the nodes. See
> [CPU Pinning & NUMA](cpu_pinning.md).

Because the gateway roles are mutually exclusive on `ai_gateway_provider`, `kserve`
depends on both: whichever one is enabled runs first, and the other is skipped.

## Two gateways, two concerns

| Gateway | Type | Purpose |
|---------|------|---------|
| `eg-gateway` | LoadBalancer (external IP) | TLS termination, hostname routing, JWT validation |
| `ai-gateway` | ClusterIP (internal only) | Model routing via header match, EPP scheduling or direct routing |

`eg-gateway` carries a deny-by-default Keycloak JWT policy; routes that authenticate
themselves (the LiteLLM API in `litellm_vkey` mode, the Langfuse UI) attach their own
no-auth `SecurityPolicy`. In **jwt** mode, `eg-gateway` validates the JWT and forwards
straight to `ai-gateway`; LiteLLM is not in the path. In **litellm_vkey** mode,
`eg-gateway` routes to LiteLLM, which validates the virtual key and then forwards to
`ai-gateway`.

```
                    EXTERNAL                            INTERNAL (cluster)

  jwt mode (default)
  Client ──HTTPS──► eg-gateway ─────────HTTP────────► ai-gateway ──► vLLM / OVMS
                    (TLS + JWT)                        (model routing)   (model)

  litellm_vkey mode
  Client ──HTTPS──► eg-gateway ──► LiteLLM ──HTTP──► ai-gateway ──► vLLM / OVMS
                    (TLS term)     (virtual keys,     (model routing)   (model)
                                    budgets, cache)
                                        │
                                        ├──► Valkey    (response cache)
                                        └──► Langfuse  (traces, token usage)
```

Step-by-step packet detail, including a verified network trace, is in
[Inference Request Flow](request_flow.md).

## Integration with the solutions repo

The core installer clones this repo to `ext/enterprise.ai-inference/` — the branch is
set in the solutions repo's `config/repos.yaml` — and wires it in automatically:

```
enterprise-ai-solutions/
├── es_auto_installer.sh          # main entry point
├── model-manager                 # thin wrapper → ext/.../model_manager/model-manager
├── config/repos.yaml             # declares this repo, and which branch to clone
├── env/<env>/global_config.yaml     # platform settings (base_domain_name, …)
├── env/<env>/config.inference.yaml  # inference settings (auth_type, …), seeded from config.yaml
├── env/<env>/models.yaml            # per-environment model catalog (seeded by `init`)
└── ext/
    └── enterprise.ai-inference/  # ← this repo
        ├── components.yaml       # component registry, merged into the core registry
        ├── config.yaml           # seed for env/<env>/config.inference.yaml
        ├── roles/                # discovered via roles_path
        ├── charts/               # vendored Helm charts (LiteLLM)
        └── model_manager/        # the model-manager CLI + templates
```

1. At preflight, `components.yaml` is auto-discovered and merged into the core component
   registry. The Jinja in each `enabled:` expression resolves before resolution.
2. `install inference` runs the enabled roles here in dependency order.
3. `env/<env>/config.inference.yaml` is loaded as extra vars, overriding role defaults.
4. After install, `./model-manager` operates standalone against the live cluster, reading
   the environment's `models.yaml` catalog.

## Model objects

`model-manager` creates standard Kubernetes CRDs, so `kubectl` manages them after
deploy:

| Runtime | Object | Routing |
|---|---|---|
| vLLM | `LLMInferenceService` | `epp` — EPP endpoint picker |
| OpenVINO | `InferenceService` | `aigateway` |

Both land in `llm_services_namespace` (default `llm-inference`).

## Related

- [Inference Request Flow](request_flow.md) — end-to-end routing for both auth modes
- [CPU Pinning & NUMA](cpu_pinning.md) — the three CPU policies and the capacity check
- [Database Architecture](database.md) — shared PostgreSQL for LiteLLM and Langfuse
- [Configuration](../customize/configuration.md) — every setting in `config.yaml`
