# Accessing Models

Both auth modes expose the **same OpenAI-compatible API** on a single endpoint: the
`model` field in the request body selects which deployed model answers. What differs
is the hostname and the credential.

`auth_type` selects the mode — `jwt` (default) or `litellm_vkey`. It is set in the
environment's `env/<env>/config.inference.yaml` (seeded from this repo's `config.yaml`).
Keycloak is deployed in both modes and people always sign in through it; `auth_type`
only selects the credential API clients send.

| | **Keycloak JWT** (`auth_type: jwt`) | **LiteLLM virtual key** (`auth_type: litellm_vkey`) |
|---|---|---|
| **Hostname** | `inference.<base_domain_name>` | `litellm.<base_domain_name>` |
| **Endpoint** | `https://inference.<domain>/v1/chat/completions` | `https://litellm.<domain>/v1/chat/completions` |
| **Credential** | Keycloak OIDC JWT (`Authorization: Bearer <jwt>`) | LiteLLM master or virtual key (`Authorization: Bearer sk-...`) |
| **Validated by** | The edge gateway, via `SecurityPolicy.jwt` | LiteLLM itself, before the request reaches a model |
| **Also deployed** | Keycloak realm + client; LiteLLM, Langfuse, Valkey with `litellm_enabled: true` | Keycloak realm + client, LiteLLM, Langfuse, Valkey |
| **Model selection** | `"model"` field in the request body | `"model"` field in the request body |

> [!IMPORTANT]
> The two modes serve on **different hostnames**. In `litellm_vkey` mode nothing is bound to
> `inference.<domain>` — requests there return `404`. Use `litellm.<domain>`.
> In `jwt` mode with `litellm_enabled: true`, `litellm.<domain>` also needs a Keycloak JWT.

Both hostnames resolve to the same edge gateway LoadBalancer IP:

```bash
export KUBECONFIG=$(pwd)/env/local/kubeconfig.yaml

GATEWAY_IP=$(kubectl get svc -n envoy-gateway-system \
  -l gateway.envoyproxy.io/owning-gateway-name=eg-gateway \
  -o jsonpath='{.items[0].status.loadBalancer.ingress[0].ip}')

DOMAIN=$(yq '.base_domain_name' env/local/global_config.yaml)
```

The examples below use `curl --resolve` so the request carries the correct `Host` and
SNI without editing `/etc/hosts`. The installer's self-signed CA is at
`env/local/logs/ai-solutions-ca.crt` — pass `--cacert <that path>` to verify the chain
properly instead of using `-k`.

## Keycloak JWT mode (default)

A helper script discovers the gateway, reads the Keycloak client credentials from the
cluster, and exchanges them for a JWT:

```bash
source ./ext/enterprise.ai-inference/model_manager/scripts/get-keycloak-token.sh
# Exports: TOKEN, GATEWAY_IP, GATEWAY_DOMAIN
```

Call the model:

```bash
curl -sk --noproxy '*' \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  --resolve "inference.${DOMAIN}:443:${GATEWAY_IP}" \
  -d '{"model":"qwen3-0-6b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}' \
  "https://inference.${DOMAIN}/v1/chat/completions"
```

List the models the gateway will route to:

```bash
curl -sk --noproxy '*' \
  -H "Authorization: Bearer $TOKEN" \
  --resolve "inference.${DOMAIN}:443:${GATEWAY_IP}" \
  "https://inference.${DOMAIN}/v1/models"
```

> [!NOTE]
> Tokens expire after 15 minutes by default (`keycloak_token_lifespan`). Re-run the
> `source` command for a fresh one, or pass `--lifespan 3600` for a 1-hour token.

A request with a missing, malformed, or expired JWT is rejected with `401` at the
gateway, before it reaches a model.

## LiteLLM virtual key mode

LiteLLM is the auth boundary. The edge gateway terminates TLS and forwards to LiteLLM,
which validates the bearer key, enforces that key's model allow-list, budget, and rate
limit, then routes on to the AI gateway.

Retrieve the master key:

```bash
MASTER_KEY=$(kubectl get secret litellm-master-key -n litellm \
  -o jsonpath='{.data.master_key}' | base64 -d)
```

Call the model:

```bash
curl -sk --noproxy '*' \
  -H "Authorization: Bearer $MASTER_KEY" \
  -H "Content-Type: application/json" \
  --resolve "litellm.${DOMAIN}:443:${GATEWAY_IP}" \
  -d '{"model":"qwen3-0-6b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}' \
  "https://litellm.${DOMAIN}/v1/chat/completions"
```

List the models registered with LiteLLM — every model that `model-manager deploy`
registered appears here:

```bash
curl -sk --noproxy '*' \
  -H "Authorization: Bearer $MASTER_KEY" \
  --resolve "litellm.${DOMAIN}:443:${GATEWAY_IP}" \
  "https://litellm.${DOMAIN}/v1/models"
```

### Virtual keys

The master key is a full-admin credential. For applications, mint a scoped virtual key
instead — with its own model allow-list, budget, and expiry:

```bash
# Create — returns {"key": "sk-..."}
VIRTUAL_KEY=$(curl -sk --noproxy '*' \
  --resolve "litellm.${DOMAIN}:443:${GATEWAY_IP}" \
  "https://litellm.${DOMAIN}/key/generate" \
  -H "Authorization: Bearer ${MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"models":["qwen3-0-6b"],"key_alias":"team-a","max_budget":5,"duration":"24h"}' \
  | jq -r '.key')
```

Use the virtual key exactly like the master key — swap `$MASTER_KEY` for `$VIRTUAL_KEY`:

```bash
curl -sk --noproxy '*' \
  -H "Authorization: Bearer $VIRTUAL_KEY" \
  -H "Content-Type: application/json" \
  --resolve "litellm.${DOMAIN}:443:${GATEWAY_IP}" \
  -d '{"model":"qwen3-0-6b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}' \
  "https://litellm.${DOMAIN}/v1/chat/completions"
```

A request with an unknown or revoked key is rejected with `401` by LiteLLM before
anything reaches a model. A key used for a model outside its allow-list is rejected too.

#### Manage keys via API

```bash
# List (includes alias, allow-list, budget, and spend to date)
curl -sk --noproxy '*' \
  --resolve "litellm.${DOMAIN}:443:${GATEWAY_IP}" \
  "https://litellm.${DOMAIN}/key/list?return_full_object=true" \
  -H "Authorization: Bearer ${MASTER_KEY}"

# Update (e.g. extend budget)
curl -sk --noproxy '*' \
  --resolve "litellm.${DOMAIN}:443:${GATEWAY_IP}" \
  "https://litellm.${DOMAIN}/key/update" \
  -H "Authorization: Bearer ${MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"key":"sk-...","max_budget":20}'

# Revoke
curl -sk --noproxy '*' \
  --resolve "litellm.${DOMAIN}:443:${GATEWAY_IP}" \
  "https://litellm.${DOMAIN}/key/delete" \
  -H "Authorization: Bearer ${MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"keys":["sk-..."]}'
```

#### Manage keys via UI

Open `https://litellm.<domain>/ui/` (login: username `admin`, password = master key) →
**Virtual Keys** tab. The UI lets you create, view spend, and revoke keys without curl.

### Observability

Request traces, token usage, and cost land in Langfuse automatically:

| UI | URL | Login |
|---|---|---|
| Langfuse | `https://langfuse.<domain>` | `langfuse_admin_email` + `LANGFUSE_ADMIN_PASSWORD` |

## Streaming

Both modes support server-sent event streaming — add `"stream": true`:

```bash
curl -sk --noproxy '*' -N \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  --resolve "inference.${DOMAIN}:443:${GATEWAY_IP}" \
  -d '{"model":"qwen3-0-6b","messages":[{"role":"user","content":"Count to five."}],"stream":true}' \
  "https://inference.${DOMAIN}/v1/chat/completions"
```

## Other endpoints

Embedding and reranking models deployed through this layer serve on the same host:

| Workload | Endpoint |
|---|---|
| Chat / completion (`llm`, `vlm`) | `/v1/chat/completions`, `/v1/completions` |
| Embedding (`embed`) | `/v1/embeddings` |
| Reranking (`rerank`) | `/v1/score`, `/v1/rerank` |

## From an SDK

Any OpenAI-compatible client works — only `base_url` and `api_key` change:

```python
from openai import OpenAI

client = OpenAI(
    # auth_type jwt:          https://inference.<base_domain_name>/v1
    # auth_type litellm_vkey: https://litellm.<base_domain_name>/v1
    base_url="https://inference.solutions.ai/v1",
    api_key="<KEYCLOAK_JWT_OR_LITELLM_KEY>",
)

print(client.chat.completions.create(
    model="qwen3-0-6b",
    messages=[{"role": "user", "content": "Hello!"}],
).choices[0].message.content)
```

For the full path a request takes through the gateways, see
[Inference Request Flow](../reference/request_flow.md).
