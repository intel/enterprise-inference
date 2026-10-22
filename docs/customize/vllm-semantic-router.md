# vLLM Semantic Router

vLLM Semantic Router is a signal-driven decision routing framework for
Mixture-of-Models deployments. It routes every request to the best model
across local, private, and frontier backends -- guided by cost, latency,
privacy, and safety constraints.

For more information, see [vllm-semantic-router.com](https://vllm-semantic-router.com/).

The router is opt-in (`vllm_sr_enabled`, off by default) and is deployed from the
upstream chart vendored at [`charts/vllm-sr`](../../charts/vllm-sr) (see its `NOTICE`).
It plugs into Envoy AI Gateway (`AIGatewayRoute`, an ext_proc filter applied through an
`EnvoyPatchPolicy`), so it runs only with `ai_gateway_provider: envoy`; with
`agentgateway` the component is skipped.

---

## Architecture

```
Client
    |
    |  HTTPS (external)
    v
LiteLLM gateway (auth_type: litellm_vkey)
    |  - authenticates the request
    |  - model="<vllm_sr_litellm_model_name>" triggers routing
    |  - starts the Langfuse trace
    v
semantic-router Gateway (dedicated Envoy gateway)
    |  - ext_proc calls the vLLM Semantic Router
    |  - router extracts semantic signals from the prompt
    |  - evaluates routing decisions and selects the best model
    v
Model backend (the model's InferencePool, deployed with model-manager)
    |  - runs inference
    |  - returns response
    v
Response flows back through the chain to the client
```

The router runs on its own `semantic-router` gateway, so normal model traffic on the
AI gateway never passes through the classifier. Using the router is a per-request
choice: call the router's model name, or call a model directly to bypass it.

With `auth_type: jwt` there is no LiteLLM in the path (`vllm_sr_expose: envoy`). The
router is reached in-cluster at
`http://semantic-router.envoy-gateway-system.svc.cluster.local/v1` with
`"model": "auto"` — the Service runs beside the Envoy Gateway controller, so
`envoy-gateway-system` is that namespace when the platform uses an existing Envoy Gateway
elsewhere. `vllm_sr_expose` (`envoy` | `litellm` | `both`) overrides the choice.

---

## Getting Started

### Prerequisites

- The inference layer is installed with `ai_gateway_provider: envoy`, with at least one
  model deployed
- The models the router may pick are listed in `vllm_sr_route_models`
  (default: the solution's default model)
- Envoy Gateway runs with `EnvoyPatchPolicy` enabled
  (`extensionApis.enableEnvoyPatchPolicy`). The kserve role turns it on together with the
  AI Gateway extension; for an Envoy Gateway the platform did not install, set
  `envoy_gateway_configure_existing: true`. The install only checks it and stops with that
  hint when it is off.

### Deploy

Set the flag in `env/<env>/config.inference.yaml`:

```yaml
vllm_sr_enabled: true
```

Then run the installer:

```bash
./es_auto_installer.sh install inference   # or: install agent-toolkit
```

The install will:
1. Deploy the vLLM Semantic Router chart into `vllm-semantic-router-system`
2. Wait for the router to become ready
3. Create the `semantic-router` gateway and its per-model routes
4. With the dashboard on, publish it on the platform gateway
5. With `auth_type: litellm_vkey`, register the router as a model in LiteLLM
   (`vllm_sr_litellm_model_name`, default `semantic-router`; the agent toolkit sets
   `vllm-sr/auto`)

`./es_auto_installer.sh validate inference` checks the router, the gateway, the
`EnvoyPatchPolicy` and, with the dashboard on, its route and auth exemption.

### Verify

Send a request through the routing layer, on the LiteLLM API
(`https://litellm.<base_domain_name>/v1/...`, or `https://<base_domain_name>/litellm/v1/...`
with `routing_mode: path`; `DOMAIN` and `GATEWAY_IP` as in
[Accessing Models](../deploy/accessing_models.md)):

```bash
curl -sk --noproxy '*' \
  --resolve "litellm.${DOMAIN}:443:${GATEWAY_IP}" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <LITELLM_KEY>" \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "write a hello world in python"}]
  }' \
  "https://litellm.${DOMAIN}/v1/chat/completions"
```

You can also call a model directly (bypasses routing):

```bash
curl -sk --noproxy '*' \
  --resolve "litellm.${DOMAIN}:443:${GATEWAY_IP}" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <LITELLM_KEY>" \
  -d '{
    "model": "qwen2-5-coder-14b",
    "messages": [{"role": "user", "content": "hello"}]
  }' \
  "https://litellm.${DOMAIN}/v1/chat/completions"
```

### Access the Dashboard

Enable the dashboard with `vllm_sr_dashboard_enabled: true` (the agent toolkit turns it
on). It is published through the platform gateway on its own host
(`vllm_sr_dashboard_hostname`):

```
https://vllm-sr.<base_domain_name>/
```

The dashboard serves root paths (`/login`, `/api`, `/assets`, ...) and cannot
run under a sub-path, so it keeps its own host with `routing_mode: path` as well. It has
its own login, so its route is exempt from the gateway's Keycloak JWT policy. The router
API itself is not exposed outside the cluster. With `vllm_sr_gateway_publish: false` (it
follows the platform's gateway flag) the dashboard is deployed but not published; reach it
with `kubectl -n vllm-semantic-router-system port-forward svc/semantic-router-dashboard 8700`.
Turning the dashboard off, or teardown, unpublishes it.

With `enforce_pss: true` the namespace enforces the `restricted` Pod Security level, or
`baseline` while the dashboard is on: the dashboard image starts as root and drops to a
non-root user itself, which `restricted` rejects.

Prerequisites for the dashboard host:

- **DNS:** `vllm-sr.<base_domain_name>` must resolve to the gateway (DNS record, or
  `/etc/hosts` on every client machine).
- **TLS certificate:** with `gateway_tls_mode: selfsigned` the platform covers it (a
  certificate of its own, or the wildcard one when services share the wildcard listener),
  chained to the platform CA (`env/<env>/logs/ai-solutions-ca.crt`). A custom certificate
  must include the host.
- **Corporate proxy:** add `vllm-sr.<base_domain_name>` to the browser/OS proxy bypass
  list, as for the other gateway hosts.

Credentials are generated at install time and stored in the `vllm-sr-dashboard` Secret:

```bash
kubectl -n vllm-semantic-router-system get secret vllm-sr-dashboard \
  -o jsonpath='{.data.admin-email}' | base64 -d; echo
kubectl -n vllm-semantic-router-system get secret vllm-sr-dashboard \
  -o jsonpath='{.data.admin-password}' | base64 -d; echo
```

---

## Request Flow

1. Client sends `POST /v1/chat/completions` with the router's model name
2. LiteLLM authenticates the request, starts a Langfuse trace, and forwards
   to the `semantic-router` gateway
3. vLLM-SR runs semantic signal classification on the prompt
4. The decision engine evaluates routing rules top-down by priority. The default
   configuration has a single `default-route` that sends every request to
   `vllm_sr_default_model`; add your own decisions (see
   [Routing Configuration](#routing-configuration))
5. The gateway routes the request to the selected model's InferencePool
6. The response flows back to the client

There is no infinite loop because LiteLLM treats the router's model name and the real
model names as separate routing entries. Only the router's model name goes to vLLM-SR;
real model names go directly to their respective backends.

---

## Configuration

### Routing Configuration

The routing config is rendered from
[`roles/vllm_sr/templates/values.yaml.j2`](../../roles/vllm_sr/templates/values.yaml.j2).
To replace it, set `vllm_sr_config_override` in `env/<env>/config.inference.yaml` to a
full routing config.

Key sections:

**Signals** define how prompts are classified:

```yaml
routing:
  signals:
    embeddings:
      - name: coding_tasks
        threshold: 0.72
        candidates:
          - "write code"
          - "implement function"
          - "debug this"
```

To add a new signal category, add another embedding entry with a name,
threshold (similarity score cutoff), and a list of candidate phrases that
represent that category.

**Decisions** define routing rules evaluated top-down by priority:

```yaml
routing:
  decisions:
    - name: coding-route
      description: "Route coding tasks to coder model"
      priority: 10
      rules:
        operator: AND
        conditions:
          - type: embedding
            name: coding_tasks
      modelRefs:
        - model: qwen2-5-coder-14b
          use_reasoning: false
```

Lower priority number means higher precedence. The first matching decision wins.

**Models** are the models the router can route to. They come from
`vllm_sr_route_models` (catalog model names, as deployed with model-manager). Each one
is listed as a model card and gets a route on the `semantic-router` gateway to its
InferencePool (`<model>-inference-pool`).

### Settings

Set these in `env/<env>/config.inference.yaml`:

| Setting | Description |
|-------|-------------|
| `vllm_sr_enabled` | Deploy the router (default `false`; needs `ai_gateway_provider: envoy`) |
| `vllm_sr_expose` | `envoy`, `litellm` or `both` (default: `litellm` with `auth_type: litellm_vkey`, else `envoy`) |
| `vllm_sr_route_models` | Deployed models the router may pick from |
| `vllm_sr_default_model` | Model served when no other decision matches |
| `vllm_sr_litellm_model_name` | Client-facing model name in LiteLLM (default `semantic-router`) |
| `vllm_sr_config_override` | Full routing config (signals, decisions), replacing the default |
| `vllm_sr_cache_backend` | Semantic cache backend: `memory` or `redis` |
| `vllm_sr_image_tag` | Router image tag (empty = the chart's `appVersion`) |
| `vllm_sr_dashboard_enabled` | Deploy the dashboard (default `false`) |
| `vllm_sr_dashboard_admin_email` | Dashboard login email (default `admin@<base_domain_name>`) |
| `vllm_sr_dashboard_hostname` | Dashboard host (default `vllm-sr.<base_domain_name>`) |
| `vllm_sr_gateway_publish` | Publish the dashboard on the platform gateway (default: the platform's gateway flag) |

Trace export from the router is disabled in this deployment; the router exposes
Prometheus metrics on port 9190 (Service `semantic-router-metrics`).

### Updating the Routing Config

After editing the settings, re-run the install:

```bash
./es_auto_installer.sh install inference
```

The Helm release is upgraded in place and the router restarts when its config changes.

---

## Onboarding a New Model

1. Deploy the model with model-manager:

   ```bash
   cd model_manager
   ./model-manager deploy <model> --wait
   ```

2. Add it to `vllm_sr_route_models`, and to a decision in `vllm_sr_config_override`
   if it should receive a specific category of prompts.
3. Re-run `./es_auto_installer.sh install inference`.

The model then shows up in the dashboard and becomes a candidate for routing.

To route a new category of prompts (for example, math or legal), add a signal
and a matching decision as described in
[Routing Configuration](#routing-configuration).

### Verify routing

Send a prompt and check which model answered (the `model` field of the response):

```bash
curl -sk --noproxy '*' \
  --resolve "litellm.${DOMAIN}:443:${GATEWAY_IP}" \
  -H "Authorization: Bearer <LITELLM_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model": "vllm-sr/auto", "messages": [{"role": "user", "content": "write a python function to sort a list"}]}' \
  "https://litellm.${DOMAIN}/v1/chat/completions" | jq -r .model
```

Clients keep calling the router's model name as before — the router dispatches to the
new model internally, so no client changes are needed.

---

## LiteLLM Integration

With `vllm_sr_expose: litellm` (the default with `auth_type: litellm_vkey`), vLLM-SR is
registered in LiteLLM as a model called `vllm_sr_litellm_model_name`. This registration
happens automatically at the end of the install, and teardown removes it. A re-run keeps an
entry that already points at the router and replaces one that does not.

Clients choose between:
- `model: "<vllm_sr_litellm_model_name>"` -- request goes through semantic routing
- `model: "<model-name>"` -- request goes directly to that model

Both paths go through LiteLLM for authentication and spend tracking.

---

## Langfuse Integration

LiteLLM records each routed request in Langfuse as it does for any model: input,
output, token counts and total latency. The router's own trace export is disabled in
this deployment.

---

## Troubleshooting

**Pod is not becoming Ready**

```bash
kubectl get pods -n vllm-semantic-router-system
kubectl logs -n vllm-semantic-router-system deploy/semantic-router
```

Common cause: on first start the router downloads its classifier models. They are
cached on a PVC, so later restarts are faster.

**Install stops at "does not enable EnvoyPatchPolicy"**

Envoy Gateway ignores the ext_proc `EnvoyPatchPolicy` until
`extensionApis.enableEnvoyPatchPolicy` is set. Re-run `install inference` so the kserve
role applies the AI Gateway extension (with `envoy_gateway_configure_existing: true` for an
Envoy Gateway the platform did not install), or set it on that Envoy Gateway release.

**LiteLLM returns "Connection error" for the router model**

Verify the `semantic-router` gateway is reachable from the LiteLLM pod (replace
`envoy-gateway-system` with the Envoy Gateway namespace if it differs):

```bash
kubectl exec -n litellm deploy/litellm -c litellm -- \
  python3 -c "import socket; s=socket.socket(); s.connect(('semantic-router.envoy-gateway-system.svc.cluster.local', 80)); print('OK')"
```

**Dashboard login fails or the page does not load**

Check the dashboard logs:

```bash
kubectl logs -n vllm-semantic-router-system deploy/semantic-router-dashboard
```
