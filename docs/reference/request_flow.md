# Inference Request Flow

End-to-end path for an inference request through the platform, from external client to model response.

## Architecture

Two auth modes are supported. The flow differs based on `auth_type` in `env/<env>/config.inference.yaml`. Keycloak is deployed in both modes for sign-in; `auth_type` only selects the API client credential:

### `jwt` (default) — Unified Routing

```
                         EXTERNAL                          INTERNAL (cluster)

 Client ──HTTPS──► eg-gateway ──HTTP──► ai-gateway ──► Model (vLLM / OVMS)
                   (TLS term)           (EPP or direct)
                   (JWT auth via
                    Keycloak OIDC)
```

Single endpoint: `https://inference.<domain>/v1/chat/completions`
Model selected by `"model": "<name>"` in the JSON body (Envoy AI Gateway extracts it automatically).

### `litellm_vkey` mode

```
                         EXTERNAL                          INTERNAL (cluster)

 Client ──HTTPS──► eg-gateway ──HTTP──► LiteLLM ──HTTP──► ai-gateway ──► vLLM
                   (TLS term)           (auth,cache)       (EPP sched)   (model)
                                            │
                                            ├──► Valkey (response cache)
                                            └──► Langfuse (traces)
```

## Two Gateways, Two Concerns

| Gateway | Type | Purpose | GatewayClass |
|---------|------|---------|--------------|
| `eg-gateway` | LoadBalancer (external IP) | TLS termination, hostname routing, JWT validation | `eg` |
| `ai-gateway` | ClusterIP (internal only) | Model routing via header match, EPP scheduling or direct routing | `eg` |

`eg-gateway` carries a deny-by-default Keycloak JWT policy. Routes that authenticate themselves (the LiteLLM API in `litellm_vkey` mode, the Langfuse UI) attach their own no-auth `SecurityPolicy`.

In `jwt` mode, `eg-gateway` validates the JWT and forwards directly to `ai-gateway`. No LiteLLM in the path.
In `litellm_vkey` mode, `eg-gateway` routes to LiteLLM (which validates the virtual key), and LiteLLM forwards to `ai-gateway`.

## Request Flow — `jwt` + Unified Routing (Default)

### Example: `qwen3-0-6b` (LLMInferenceService, EPP routing)

---

### Step 1: Client → eg-gateway (TLS + JWT Validation)

```
curl https://inference.example.com/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-0-6b", "messages": [{"role":"user","content":"Hello"}]}'
```

- Client connects to `192.168.120.103:443` (eg-gateway LoadBalancer)
- TLS terminated at the HTTPS listener (`*.example.com`)
- Hostname `inference.example.com` matched by HTTPRoute
- SecurityPolicy validates the JWT (Keycloak OIDC issuer)
- Invalid token → HTTP 401 (request stops here)

**Kubernetes resources:**
- `Gateway/eg-gateway` in `envoy-gateway-system`
- `HTTPRoute/inference-external` in `envoy-gateway-system` → backend `ai-gateway:80`
- `SecurityPolicy/gateway-auth` on `eg-gateway` (deny-by-default JWT validation against Keycloak)

---

### Step 2: eg-gateway → ai-gateway (Model Routing)

- Request forwarded to ai-gateway ClusterIP service (port 80)
- Envoy AI Gateway extracts `"model": "qwen3-0-6b"` from the JSON body
- Sets header `x-ai-eg-model: qwen3-0-6b` internally
- AIGatewayRoute matches the header and selects the backend

**Two routing modes:**

| Mode | Backend | When |
|------|---------|------|
| `epp` | `InferencePool` → EPP scheduler picks optimal pod | LLMInferenceService (default) |
| `direct` | `Service/<model>-kserve-workload-svc:8000` | InferenceService or explicit `routing: direct` |

**Kubernetes resources:**
- `Gateway/ai-gateway` in `envoy-ai-gateway-system`
- `AIGatewayRoute/<model>-aigateway-route` (header-based match, no hostname)

---

### Step 3a: EPP Routing — ai-gateway → EPP Scheduler → vLLM

- Envoy's `ext_proc` filter invokes the EPP gRPC service (port 9002)
- EPP evaluates all available replicas: queue depth, KV cache utilization, load
- Returns the selected pod IP to Envoy
- Envoy forwards to `<pod-ip>:8000`

**Kubernetes resources:**
- `InferencePool/<model>-inference-pool` → `endpointPickerRef: <model>-epp-service:9002`

### Step 3b: Direct Routing — ai-gateway → workload service

- Request forwarded directly to `<model>-kserve-workload-svc:8000`
- For OpenVINO models: ai-gateway rewrites `/v1/` → `/v3/` (OVMS uses v3 API internally)
- No EPP involved — useful for non-LLM models or models that don't support EPP

---

### Step 4: Response Path

```
Model → ai-gateway → eg-gateway → Client
```

No caching or tracing in this path (add observability via OpenTelemetry sidecar if needed).

---

## Request Flow — `litellm_vkey` Mode

### Example: `llama3-8b-awq` model, deployed via KServe LLMInferenceService

---

### Step 1: Client → eg-gateway (TLS Termination)

```
curl https://litellm.inference-example.com/v1/chat/completions \
  -H "Authorization: Bearer sk-<virtual-key>" \
  -d '{"model": "llama3-8b-awq", "messages": [...]}'
```

- Client connects to `192.168.120.103:443` (eg-gateway LoadBalancer)
- TLS terminated at the HTTPS listener (`*.inference-example.com`)
- Hostname `litellm.inference-example.com` matched by HTTPRoute

**Kubernetes resources:**
- `Gateway/eg-gateway` in `envoy-gateway-system` (listeners: http/80, https/443)
- `HTTPRoute/litellm` in `litellm` namespace → backend `litellm:4000`
- `SecurityPolicy/litellm-no-auth` (overrides the gateway JWT policy; LiteLLM checks the key)

---

### Step 2: eg-gateway → LiteLLM (Virtual Key Auth)

- Request forwarded to LiteLLM service (`10.233.27.129:4000`)
- LiteLLM validates the virtual key (`sk-<key>`) against its database
- Invalid key → HTTP 401 (request stops here)
- Valid key → spend tracking, rate limit check, proceed

**What LiteLLM does:**
1. Authenticates the virtual key
2. Checks Redis cache (hit → return cached response, skip steps 3-5)
3. Resolves model name to backend `api_base` (from DB: `store_model_in_db: true`)
4. Forwards to the model endpoint
5. On response: writes to Langfuse (async), caches in Valkey

---

### Step 3: LiteLLM → ai-gateway (EPP Routing)

LiteLLM calls the registered `api_base`:
```
POST http://envoy-llm-inference-ai-gateway-62f06ddf.envoy-gateway-system.svc.cluster.local
     /llm-inference/llama3-8b-awq/v1/chat/completions
```

- Request arrives at ai-gateway Envoy proxy (ClusterIP `10.233.10.164:80`)
- Path `/llm-inference/llama3-8b-awq/v1/chat/completions` matches HTTPRoute rule
- Backend type is `InferencePool` (not a plain Service)

**Kubernetes resources:**
- `Gateway/ai-gateway` in `llm-inference` (GatewayClass: `inference-pool-with-aigwroute`)
- `HTTPRoute/llama3-8b-awq-kserve-route` in `llm-inference`
- `InferencePool/llama3-8b-awq-inference-pool` in `llm-inference`

---

### Step 4: ai-gateway → EPP Scheduler (ext_proc)

- Envoy's `ext_proc` filter invokes the EPP gRPC service (port 9002)
- EPP scheduler evaluates all available model replicas
- Picks the best endpoint based on: queue depth, KV cache utilization, load
- Returns the selected pod IP to the Envoy proxy

**Kubernetes resources:**
- `InferencePool` spec: `endpointPickerRef → llama3-8b-awq-epp-service:9002`
- EPP runs in pod `llama3-8b-awq-kserve-router-scheduler` (containers: `main`, `tokenizer`)

---

### Step 5: ai-gateway → vLLM Pod (Inference)

- Envoy rewrites path: `/llm-inference/llama3-8b-awq/v1/chat/completions` → `/v1/chat/completions`
- Routes to the EPP-selected pod (`10.233.104.79:8000`)
- vLLM processes the request and returns the completion

**Kubernetes resources:**
- `Deployment/llama3-8b-awq-kserve` (vLLM pods with model loaded from PVC)
- `Service/llama3-8b-awq-kserve-workload-svc` (headless, for endpoint discovery)

---

### Step 6: Response Path (reverse)

```
vLLM → ai-gateway (EPP logs response metrics) → LiteLLM → eg-gateway → Client
```

LiteLLM on response:
- Caches in Valkey (`allkeys-lru`, TTL=600s) for identical future requests
- Sends trace to Langfuse (async callback): input, output, latency, token usage, model

---

## Verified Network Trace

Actual request correlation from logs:

```
Pod IPs:
  LiteLLM:     10.233.104.75
  ai-gateway:  10.233.104.102
  EPP router:  10.233.104.80
  vLLM:        10.233.104.79

ai-gateway access log:
  downstream_remote: 10.233.104.75 (LiteLLM)        ← confirms LiteLLM is the caller
  route: rule/1 (InferencePool-backed)               ← EPP route, not catch-all
  upstream_host: 10.233.104.79:8000 (vLLM)           ← EPP picked this pod
  path rewritten: /v1/chat/completions               ← prefix stripped
  x-request-id: b918d98a-...                         ← correlates across all logs

EPP scheduler log:
  "EPP received request" x-request-id=b918d98a-...   ← same request
  "EPP sent request body response(s) to proxy"       ← picked target pod
  "EPP sent response body back to proxy"             ← processed response

Langfuse trace:
  timestamp: 2026-06-10T13:58:52Z
  input: "What color is the sky? One word."
  output: "Blue."
  latency: 0.646s
```

---

## Cache Behavior

| Request | Path | Latency |
|---------|------|---------|
| First call | LiteLLM → ai-gateway → EPP → vLLM | ~2s |
| Repeat (cached) | LiteLLM → Valkey (hit) | ~0.06s |

Cache key: model + messages hash. TTL: 600 seconds. Eviction: `allkeys-lru`.

---

## Auto-Registration (Model Manager)

When a model is deployed via `model-manager deploy`:

1. KServe controller creates `LLMInferenceService` with `router.gateway.refs → ai-gateway`
2. Controller generates HTTPRoute attached to ai-gateway + InferencePool
3. Background process (`_register_bg.py`) waits for `Ready` condition
4. Discovers ai-gateway dataplane via label: `gateway.envoyproxy.io/owning-gateway-name=ai-gateway`
5. Probes the gateway URL to confirm it's routable
6. With `auth_type: litellm_vkey`, registers the model in LiteLLM with `api_base = http://<ai-gw-dataplane>/<ns>/<model>/v1`

If the gateway isn't routable yet (e.g., EPP not configured), falls back to the direct workload service URL.

---

## Key Configuration Files

| File | Purpose |
|------|---------|
| `roles/litellm/tasks/install.yaml` | Deploys LiteLLM + Valkey subchart via Helm |
| `roles/litellm/defaults/main.yaml` | Topology-aware sizing, cache config |
| `charts/litellm-helm/` | Vendored chart with bitnami/valkey subchart |
| `model_manager/mm/litellm.py` | Auto-registration with LiteLLM API |
| `model_manager/mm/manifests.py` | LLMInferenceService manifest (gateway ref) |

---

## Failure Modes

| Failure | Behavior |
|---------|----------|
| Invalid virtual key | Rejected at LiteLLM (401), never reaches model |
| Model not registered in LiteLLM | LiteLLM returns 404 model not found |
| ai-gateway down | LiteLLM returns 503, no inference |
| EPP scheduler down | `failureMode: FailOpen` — routes to any available pod |
| vLLM pod crash | EPP stops selecting that pod (health-based routing) |
| Langfuse down | Inference still works, traces dropped silently |
| Valkey down | Inference still works, no caching (higher latency) |
