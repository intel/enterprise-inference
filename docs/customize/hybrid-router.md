# Hybrid Router — Intelligent Inference Routing for the LiteLLM Gateway

The **Hybrid Router** adds cost- and latency-aware model selection to the LiteLLM
gateway. Clients call a single virtual model (`hybrid-router`); the router
classifies each request by complexity and dispatches it to the best real model for
that tier, learning over time which model is most efficient.

It is integrated into the stock LiteLLM proxy image **without rebuilding the image** —
the router code is mounted from a Kubernetes ConfigMap and activated through a
LiteLLM callback. Enable or disable it with a single setting. It runs inside LiteLLM, so
it needs LiteLLM deployed: `auth_type: litellm_vkey`, or `litellm_enabled: true` with
Keycloak JWT auth.

---

## Table of Contents

- [Why](#why)
- [How it works](#how-it-works)
- [Two routing layers](#two-routing-layers)
  - [Layer 1 — Complexity classification](#layer-1--complexity-classification)
  - [Layer 2 — Efficiency (multi-armed bandit)](#layer-2--efficiency-multi-armed-bandit)
- [Deployment](#deployment)
- [Configuration reference](#configuration-reference)
- [Onboarding the virtual model](#onboarding-the-virtual-model)
- [Testing](#testing)
- [Observability](#observability)
- [File layout](#file-layout)
- [Disabling the router](#disabling-the-router)

---

## Why

Sending every request to a large model is slow and expensive; sending everything to
a small model sacrifices quality on hard prompts. The Hybrid Router picks the
smallest model that can handle each request:

- **Simple** prompts (greetings, factual lookups) → small, fast models.
- **Complex / reasoning** prompts (architecture, chain-of-thought, proofs) → larger models.
- Within a tier, an online **multi-armed bandit (MAB)** learns which candidate model is
  fastest and highest-throughput, and shifts traffic toward it automatically.

The result is lower average latency and cost with no client-side changes — callers
keep calling one model name.

---

## How it works

The integration mirrors LiteLLM's own built-in `ComplexityRouter`. At proxy startup,
the custom module monkey-patches `litellm.router.Router` to recognise a new
`auto_router/hybrid_router` deployment prefix. A pre-routing hook then intercepts every
request to the virtual model *before* a deployment is chosen.

```
Request  model="hybrid-router"
  │
  ▼
Router.async_pre_routing_hook  (patched by custom_hybrid_router.py)
  │  model in hybrid_routers?   NO  → normal LiteLLM routing (unaffected)
  │                             YES ↓
  │  1. Extract user + system text from messages
  │  2. HybridRouter.classify()   → SIMPLE / MEDIUM / COMPLEX / REASONING
  │  3. HybridRouter.pick_model() → best real model for that tier (MAB bandit)
  │  4. Return PreRoutingHookResponse(model=<chosen real model>)
  │
  ▼
LiteLLM routes to the chosen real model as normal
  │
  ▼
async_log_success_event / async_log_failure_event
     Compute latency + token throughput → reward → bandit.update()
     (the bandit learns, per tier, which model is fastest over time)
```

Requests to any other model name (`qwen3-4b`, `qwen2-5-coder-14b`, …) are **not touched** —
only traffic addressed to the virtual `hybrid-router` entry is intercepted.

---

## Two routing layers

### Layer 1 — Complexity classification

A lightweight, rules-based classifier (no extra model call) scores each prompt across
several dimensions — estimated token count, code / reasoning / technical / simple
keyword hits, multi-step patterns (`first … then`, `step 2`, numbered lists), and
question density — then maps the weighted score to a tier.

| Tier | Score range | Typical queries |
|------|-------------|-----------------|
| `SIMPLE` | `< 0.15` | Greetings, factual lookups, yes/no |
| `MEDIUM` | `0.15 – 0.35` | Explanations, summaries, basic coding |
| `COMPLEX` | `0.35 – 0.60` | Architecture, multi-step code, technical analysis |
| `REASONING` | `≥ 0.60` **or** 2+ reasoning markers | Chain-of-thought, deduction, proof |

Tier boundaries, token thresholds, keyword lists, and dimension weights are all
overridable via `complexity_router_config` (see the reference below).

### Layer 2 — Efficiency (multi-armed bandit)

Each tier owns an independent bandit whose arms are that tier's candidate models. A
model may be fast for `SIMPLE` prompts but slow for `COMPLEX` ones (longer generation),
so per-tier bandits learn these behaviours separately.

After each response the router computes a reward and updates the arm:

```
time_per_token = latency_seconds / max(completion_tokens, 1)
reward         = target_tpt / (target_tpt + time_per_token)     # in (0, 1]
```

Failures record a reward of `0`. When `quality_weight > 0`, the reward blends
efficiency with a quality score:
`reward = (1 - quality_weight) * efficiency + quality_weight * quality_score`.

Three bandit algorithms ship (selectable via `bandit`):

| Value | Algorithm | Behaviour |
|-------|-----------|-----------|
| `thompson` *(default)* | Beta-Bernoulli Thompson Sampling | Posterior sampling; strong exploration/exploitation balance |
| `ucb` | Upper Confidence Bound | Optimism under uncertainty; `delta` widens the exploration bound |
| `epsilon_greedy` | ε-greedy | Explores uniformly with probability `epsilon`, else greedy |

> A tier with only one candidate model skips the bandit entirely and routes directly.

---

## Deployment

The router is part of the **`litellm`** role and is controlled by one setting in
`env/<env>/config.inference.yaml`:

```yaml
hybrid_router_enabled: true
```

Then run `./es_auto_installer.sh install inference` (or `install agent-toolkit`).

The router is wired into the LiteLLM Helm release, and the install leaves an existing
release as it is. To add, change or remove the router on a stack whose LiteLLM is already
installed, upgrade the release with `litellm_reconcile`:

```bash
./es_auto_installer.sh install inference -- -e litellm_reconcile=true
```

(or set `litellm_reconcile: true` in `env/<env>/config.inference.yaml`, which upgrades the
release on every install).

When `hybrid_router_enabled: true`, the install:

1. Renders the router source into the `litellm-router-code` ConfigMap
   ([`roles/litellm/tasks/hybrid_router_setup.yaml`](../../roles/litellm/tasks/hybrid_router_setup.yaml)).
2. Volume-mounts each Python file into `/etc/litellm/` and `/etc/litellm/router_strategy/…`
   in the LiteLLM container (`hybrid_router_mount_dir`).
3. Registers the callback so LiteLLM imports the module (and applies the monkey-patch)
   at startup:

   ```yaml
   litellm_settings:
     callbacks: ["custom_hybrid_router.proxy_handler_instance"]
   ```

4. Adds the virtual `hybrid-router` model to LiteLLM's model list, built from the
   settings below.
5. Sets `router_settings.routing_strategy` to `hybrid_router_base_strategy`, with the
   LiteLLM cache (Valkey) as shared routing state when it is deployed
   (`litellm_cache_enabled`, default `true`).

No image rebuild is required. To change router *code*, edit the files under
[`roles/litellm/files/router/`](../../roles/litellm/files/router/) and re-run the
install (a code change rolls the LiteLLM pods, as mounted files are not refreshed in
place); to change router *config*, edit the settings below and re-run the install — both
with `litellm_reconcile: true` once LiteLLM is installed.

---

## Configuration reference

Set these in `env/<env>/config.inference.yaml`. They become the virtual model's
`litellm_params.hybrid_router_config`.

| Setting | Key in `hybrid_router_config` | Default | Description |
|---------|-----|---------|-------------|
| `hybrid_router_bandit` | `bandit` | `thompson` | MAB algorithm: `ucb`, `epsilon_greedy`, `thompson` |
| `hybrid_router_delta` | `delta` | `0.1` | UCB exploration parameter — smaller = more exploration (must be in `(0, 1)`) |
| `hybrid_router_epsilon` | `epsilon` | `0.1` | ε-greedy exploration probability (must be in `[0, 1]`) |
| `hybrid_router_target_tpt` | `target_tpt` | `1.0` | Target time-per-token; sets the reward curve's midpoint |
| `hybrid_router_quality_weight` | `quality_weight` | `0.0` | Quality-vs-efficiency blend in `[0, 1]`; `0` = pure efficiency |
| `hybrid_router_tier_candidates` | `tier_candidates` | `hybrid_router_default_model` for every tier | Map of tier → list of deployed model names |
| `hybrid_router_tier_boundaries` | `complexity_router_config.tier_boundaries` | `0.15`, `0.35`, `0.60` | `simple_medium`, `medium_complex`, `complex_reasoning` |
| `hybrid_router_default_model` | `hybrid_router_default_model` | the solution's default model, else `qwen3-4b` | Fallback model |
| `hybrid_router_model_name` | — | `hybrid-router` | Client-facing virtual model name |

Every model listed in `tier_candidates` / `hybrid_router_default_model` must already be
deployed and registered in LiteLLM. model-manager registers the models it deploys with
`auth_type: litellm_vkey`; with `litellm_enabled: true` add them in LiteLLM yourself.

---

## Onboarding the virtual model

The install adds the virtual entry for you. The resulting model entry looks like this:

```yaml
model_list:
  # --- virtual hybrid-router entry (what clients call) ---
  - model_name: hybrid-router
    litellm_params:
      model: auto_router/hybrid_router
      hybrid_router_default_model: qwen3-4b
      hybrid_router_config:
        bandit: thompson
        target_tpt: 1.0
        quality_weight: 0.0
        tier_candidates:
          SIMPLE:    [qwen3-4b, qwen2-5-coder-14b]
          MEDIUM:    [qwen3-4b, qwen2-5-coder-14b]
          COMPLEX:   [qwen2-5-coder-14b]
          REASONING: [qwen2-5-coder-14b]
        complexity_router_config:
          tier_boundaries:
            simple_medium: 0.15
            medium_complex: 0.35
            complex_reasoning: 0.60
```

The `auto_router/hybrid_router` model is never dispatched directly — the pre-routing
hook rewrites the target before LiteLLM selects a deployment.

### Onboarding at runtime via the `/model/new` API

LiteLLM stores models in its database (`store_model_in_db`), so you can add another
virtual entry live — no config edit or restart — by POSTing to `/model/new` with the
master key. The admin API is not published on the gateway (only the inference paths
are), so call it over a port-forward:

```bash
# Retrieve the LiteLLM master key
export LITELLM_MASTER_KEY=$(kubectl -n litellm get secret litellm-master-key \
  -o jsonpath='{.data.master_key}' | base64 -d)
kubectl -n litellm port-forward svc/litellm 4000:4000 &

curl -s --location 'http://localhost:4000/model/new' \
  --header "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
  --header 'Content-Type: application/json' \
  --data '{
    "model_name": "hybrid_router2",
    "litellm_params": {
        "model": "auto_router/hybrid_router",
        "hybrid_router_default_model": "qwen2-5-coder-14b",
        "hybrid_router_config": {
            "bandit": "thompson",
            "delta": 0.1,
            "epsilon": 0.1,
            "target_tpt": 1.0,
            "quality_weight": 0.0,
            "tier_candidates": {
                "SIMPLE": ["llama-3-1-8b", "qwen2-5-coder-14b"],
                "MEDIUM": ["llama-3-1-8b", "qwen2-5-coder-14b"],
                "COMPLEX": ["qwen3-coder-30b"],
                "REASONING": ["qwen3-coder-30b"]
            },
            "complexity_router_config": {
                "tier_boundaries": {
                    "simple_medium": 0.15,
                    "medium_complex": 0.35,
                    "complex_reasoning": 0.60
                },
                "token_thresholds": {
                    "simple": 15,
                    "complex": 400
                }
            }
        }
    },
    "model_info": {}
}'
```

The `model_name` is what clients call (here `hybrid_router2`), and every model listed
in `tier_candidates` / `hybrid_router_default_model` must already be onboarded on the
gateway.

---

## Testing

Send any request to `model="hybrid-router"` on the LiteLLM API
(`https://litellm.<base_domain_name>/v1/...`, or `https://<base_domain_name>/litellm/v1/...`
with `routing_mode: path`; `DOMAIN` and `GATEWAY_IP` as in
[Accessing Models](../deploy/accessing_models.md)):

```bash
curl -sk --noproxy '*' \
  --resolve "litellm.${DOMAIN}:443:${GATEWAY_IP}" \
  -H "Authorization: Bearer <LITELLM_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model": "hybrid-router", "messages": [{"role": "user", "content": "What is 2+2?"}]}' \
  "https://litellm.${DOMAIN}/v1/chat/completions"
```

The router classifies the prompt and transparently forwards it. Requests addressed to
any other model name are unaffected.

---

## Observability

The proxy logs one line at startup and one per routed request:

```
[HybridRouterProxy] litellm.router.Router patched for auto_router/hybrid_router
[HybridRouterProxy] Initialised model_name=hybrid-router bandit=thompson default=qwen3-4b tiers=['SIMPLE', 'MEDIUM', 'COMPLEX', 'REASONING']
[HybridRouterProxy] model=hybrid-router call_id=... tier=SIMPLE -> qwen3-4b
```

Inspect them with:

```bash
kubectl logs -n litellm deploy/litellm -c litellm -f
```

Per-tier bandit state (arm counts, running means, and — for Thompson — the Beta
`alpha`/`beta` posteriors) is exposed programmatically via
`HybridRouterProxy.router_state()`. Latency and token metrics also flow to Langfuse
through the standard LiteLLM telemetry.

---

## File layout

```
roles/litellm/
├── defaults/main.yaml                ← hybrid_router_* settings
├── tasks/hybrid_router_setup.yaml    ← ConfigMap, mounts, callback, virtual model
└── files/router/
    ├── custom_hybrid_router.py       ← callback + Router monkey-patch (entry point)
    └── router_strategy/
        ├── efficiency_router/
        │   ├── bandit.py             ← UCB / EpsilonGreedy / ThompsonSampling
        │   └── efficiency_router.py  ← standalone efficiency router + reward fns
        └── hybrid_router/
            └── hybrid_router.py      ← two-layer router (classify → pick)
```

---

## Disabling the router

Set `hybrid_router_enabled: false` and re-run the install with `litellm_reconcile: true`
(see [Deployment](#deployment)). The ConfigMap mounts, the callback and the virtual
`hybrid-router` model are dropped, then the `litellm-router-code` ConfigMap is removed, so
the gateway behaves as a stock LiteLLM proxy. Remove any virtual entries you added through
`/model/new` as well (LiteLLM UI or `/model/delete`) so clients no longer reference them.
