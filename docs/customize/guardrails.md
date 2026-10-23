# Guardrails — Safety Content Moderation

**Guardrails** provides safety content moderation for LLM requests that go through LiteLLM. It screens chat requests through a dedicated guard model before they reach the main LLM, blocking unsafe or policy-violating content. It is on by default with `auth_type: litellm_vkey`, where LiteLLM is the only way to the models.

Two components:

1. **Guard model** — a small CPU vLLM model from the catalog, selected by `guardrail_model` (default: `qwen3guard`, serving `Qwen/Qwen3Guard-Gen-0.6B`), deployed by model-manager
2. **A `custom_code` guardrail registered with LiteLLM** — the screening logic itself, sandboxed Python stored in LiteLLM and run on its `pre_call` hook. It calls the guard model directly over HTTP.

The screening logic lives in one Ansible
template, [qwen3guard_gen.py.j2](../../roles/litellm/templates/qwen3guard_gen.py.j2), which
the litellm role ([register_guardrail.yaml](../../roles/litellm/tasks/register_guardrail.yaml))
renders (guard endpoint, model id, policy) and registers through LiteLLM's `/guardrails` API.
LiteLLM compiles it with RestrictedPython and calls its `apply_guardrail()` on every request.

The guard model is an ordinary catalog entry (`qwen3guard` in
[models.yaml](../../model_manager/models.yaml)), deployed and served like any other model.

The guardrail's **identity is model-agnostic**: it is registered as `content_safety`
whatever guard model is deployed. Swapping `guardrail_model` changes only the model behind
it, never its name, so no registration is ever left stranded. The model actually in
use is baked into the registered code and shown in the guardrail's description in the
LiteLLM UI.

> **A blocked request is an error response.** LiteLLM returns **HTTP 500** with the block
> reason in `error.message`, for example
> `Custom code guardrail execution failed: Blocked by the content-safety guardrail: unsafe content (categories: Non-violent Illegal Acts).`
>
> Re-running the install is idempotent: the registration step **updates** the stored code
> via `PUT /guardrails/{id}` when it differs (and leaves an identical registration alone),
> which re-initializes the in-memory guardrail — no LiteLLM restart, so an edited template
> or a different guard model takes effect on the next request.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Request Flow                             │
└─────────────────────────────────────────────────────────────┘

  User Request
       │
       ▼
  ┌──────────────────────────────┐
  │  LiteLLM gateway             │
  │  ┌────────────────────────┐  │  pre_call hook       ┌─────────────────────────┐
  │  │ content_safety         │  │  http_post           │  Guard model            │
  │  │ (custom_code sandbox)  │──┼─────────────────────►│  (guardrail_model)      │
  │  └────────────────────────┘  │◄─────────────────────│  via the AI gateway     │
  └──────────────┬───────────────┘  Safety: Safe|Unsafe  └─────────────────────────┘
                 │                                        Namespace: llm-inference
      safe ──────┴────── unsafe
       │                   │
       ▼                   ▼
  ┌─────────────┐   HTTP 500 with the
  │  Main LLM   │   block reason in error.message
  └─────────────┘
       │
       ▼
  Response to User
```

**Namespaces:**
- `llm-inference` — the guard model (`qwen3guard`), next to the other models
- `litellm` — LiteLLM proxy; holds the registered guardrail in its DB

---

## Configuration

### Enable/Disable

Set in `env/<env>/config.inference.yaml`:

```yaml
guardrails_enabled: true   # default: true with auth_type: litellm_vkey
```

- `true` — deploys the guard model and registers the guardrail; LiteLLM screens every chat request before it reaches the main LLM.
- `false` — skips the step entirely; the stack runs unchanged with no content moderation.

The guardrail runs inside LiteLLM, so it screens only what passes through LiteLLM. It is
on by default with `auth_type: litellm_vkey`, where every API request does. With
`auth_type: jwt` clients call the models through the AI gateway directly, so it is off,
also when LiteLLM is added with `litellm_enabled: true`.

**Guard model:** `guardrail_model` (default `qwen3guard`) is the catalog name of the guard
model. It selects the model the guardrail calls; the code template that parses its verdicts
is `qwen3guard_gen.py.j2`.

| `guardrail_model` value | Model | Code template | Status |
|---|---|---|---|
| `qwen3guard` | `Qwen/Qwen3Guard-Gen-0.6B` | `qwen3guard_gen.py.j2` | Supported (default) |

**Fail-closed behaviour:** requests are blocked when the guard is unreachable, when it
returns a non-2xx, when the prompt exceeds its context window, or when its reply cannot be
parsed. An unparseable verdict is never treated as "safe" — otherwise a change in the
model's output format would silently disable moderation while everything looked healthy.
Each case produces a distinct block message, so the cause is visible in the response.

### Guard endpoint and policy

All of these are settings, substituted into the code at install time — the template
itself never needs editing:

| Setting | Default | Purpose |
|---|---|---|
| `guard_service_url` (alias `guardrail_service_url`) | derived, see below | Guard endpoint, through the in-cluster AI gateway. Override to screen against an OpenAI-compatible guard elsewhere (another namespace, off-cluster). |
| `guard_block_on_controversial` | `false` | Also block Qwen3Guard's `Controversial` label. Off by default: it is advisory and blocking it rejects a lot of legitimate traffic. |
| `guard_max_tokens` | `32` | Generation cap for the verdict. Verdicts are a few tokens; a larger cap only lengthens the blocking hook. |
| `guard_timeout` | `60` | Guard call timeout (LiteLLM clamps this to 60s). |

The derived endpoint is the AI gateway's data-plane Service for the active
`ai_gateway_provider`, followed by the guard model's route
(`/<llm_services_namespace>/<guardrail_model>/v1/chat/completions`):

| `ai_gateway_provider` | `guard_service_url` (defaults) |
|---|---|
| `envoy` | `http://ai-gateway.envoy-gateway-system.svc.cluster.local/llm-inference/qwen3guard/v1/chat/completions` — the Service runs beside the Envoy Gateway controller, so `envoy-gateway-system` is replaced by that namespace when the platform uses an existing Envoy Gateway elsewhere |
| `agentgateway` | `http://ai-gateway.llm-inference.svc.cluster.local/llm-inference/qwen3guard/v1/chat/completions` — the Service runs in the agentgateway gateway namespace (`ai_gateway_namespace`, the model namespace by default) |

```yaml
# e.g. point the guardrail at a guard model in another namespace
# env/<env>/config.inference.yaml
guard_service_url: "http://ai-gateway.envoy-gateway-system.svc.cluster.local/other-ns/qwen3guard/v1/chat/completions"
```

Then re-run `./es_auto_installer.sh install inference` — the install upgrades LiteLLM so
the new guard host reaches its allowlist (next section).

### Outbound URL allowlist

LiteLLM checks every URL a guardrail calls and refuses private, cluster-internal and
loopback addresses unless the host is allowlisted — the guard endpoint is a ClusterIP
Service, so without the allowlist every request would fail closed with
`the guard model backend is unreachable (Blocked URL: URL targets a blocked address ...)`. While the guardrail
is on, the install adds exactly the host of `guard_service_url` (and `host:port` when the
URL names a port) to `litellm_settings.user_url_allowed_hosts` in the LiteLLM config; the
SSRF check stays on for every other host. Do not set `user_url_validation: false`, which
turns the check off for every URL.

LiteLLM sends the allowlisted call to the address the guard host resolves to. Behind an
HTTP proxy that address matches neither the `.svc` / `.cluster.local` suffixes nor the CIDR
entries of `no_proxy` (the HTTP client does not honour CIDRs), so the call would leave
through the proxy and come back as `the guard model backend is unreachable (403)`. While
the guardrail is on and a proxy is set, the install adds the ClusterIP of the guard
Service to the LiteLLM pod's `no_proxy`.

The allowlist and the pod's `no_proxy` are part of the LiteLLM Helm release. An existing
release is otherwise left as it is, but one whose running config (ConfigMap `litellm-config`)
lacks the guard host, or whose pod `no_proxy` lacks the guard address, is upgraded by the
install with the current settings — a release installed before the guardrail, or after
`guard_service_url` changed, needs no extra flag. The registration step checks the running
config again and fails instead of registering a guardrail that would block everything.

### Resource Allocation

Edit the `qwen3guard` entry in [models.yaml](../../model_manager/models.yaml) (or in the
environment's copy, `env/<env>/models.yaml`, which the install prefers):

```yaml
  - name: qwen3guard
    model_id: Qwen/Qwen3Guard-Gen-0.6B
    cpu: 8         # CPUs for the guard model
    memory: 16Gi   # Memory
    args: [ "--block-size", "128", "--dtype", "bfloat16", "--max-num-seqs", "8", "--tensor-parallel-size", "1" ]
    env:
      VLLM_CPU_KVCACHE_SPACE: "4"
```

**8Gi is not enough** — the guard's four vLLM processes peak at
~8.6 GiB (a 6.8 GiB worker holding the 4 GiB KV cache plus bf16 weights, a ~0.9 GiB API
server, a ~0.7 GiB EngineCore), so an 8Gi limit OOM-kills the pod with exit code 137
in a `CrashLoopBackOff`. Because the guardrail is fail-closed, a guard OOM blocks every
LiteLLM request.
Do *not* cut `VLLM_CPU_KVCACHE_SPACE` to fit instead: at 4 GiB it yields 37,376 tokens against
a 32,768 context window, and halving it puts the cache below the guard's own window, so long
prompts fail closed rather than being screened.

### Context window

`--max-model-len` is deliberately **not** set for the guard model, so it inherits the guard
model's own `max_position_embeddings` — the same way the served models are configured.
Both sides therefore track their own model instead of a literal that has to be kept in sync
by hand.

This matters because the guardrail runs `pre_call` and is handed the *same* prompt the
served LLM is about to receive. If the guard's context window is the smaller of the two,
prompts in the gap are rejected by the guard and blocked (fail-closed) rather than screened
— the guardrail silently becomes the request-size ceiling for the whole stack.

Current pairing with the agent toolkit's default model (both 32768, so no gap):

| | source of context length | effective |
|---|---|---|
| served LLM `qwen2-5-coder-14b` (`Qwen/Qwen2.5-Coder-14B-Instruct`) | model default | 32768 |
| guard `qwen3guard` (`Qwen/Qwen3Guard-Gen-0.6B`) | model default | 32768 |

Verify after any model change:

```bash
for m in qwen2-5-coder-14b qwen3guard; do
  kubectl exec -n llm-inference deploy/$m-kserve -c main -- python3 -c \
    "import json,urllib.request; print([m['max_model_len'] for m in json.load(urllib.request.urlopen('http://localhost:8000/v1/models'))['data']])"
done
```

If the guard's number is lower, over-length requests fail closed with `the prompt is longer
than the guard model's context window` (reported distinctly from an unreachable guard).
Resolving it needs a deliberate choice — a longer-context guard model, or chunked screening
— not a larger `--max-model-len`, which would only make vLLM accept a prompt the model
cannot actually attend over.

### Adding a guard model

1. Add the guard model to the catalog ([models.yaml](../../model_manager/models.yaml)).
2. If it reports verdicts in Qwen3Guard's `Safety:` / `Categories:` format, reuse the
   existing template. Otherwise add `roles/litellm/templates/<family>.py.j2` — copy the
   existing one and change the prompt/parsing — and point
   [register_guardrail.yaml](../../roles/litellm/tasks/register_guardrail.yaml) at it.
   Keep the sandbox rules (see the template's docstring): no `import`, no `class`, no
   leading-underscore names, no f-strings; use the injected primitives (`allow`, `block`,
   `trim`, `lower`, `contains`, `json_parse`, `json_stringify`, `regex_find_all`, `http_post`).
3. Add a row to the table above, set `guardrail_model`, re-run the install. The endpoint
   and model id are substituted for you.

The install compiles the rendered code in LiteLLM's sandbox
(`POST /guardrails/test_custom_code`) and fails **before** registering if it is rejected, so
a sandbox mistake cannot take the guardrail down.

---

## Deployment

### Prerequisites

- `auth_type: litellm_vkey` in `env/<env>/config.inference.yaml`; LiteLLM holds the
  guardrail registration
- The guard model in the model catalog — `env/<env>/models.yaml` when it exists, or the
  shipped [models.yaml](../../model_manager/models.yaml) when that copy (seeded at `init`)
  lacks the `qwen3guard` entry; a name in neither fails the install
- Free CPU/memory for the guard model (see `cpu`/`memory` on the `qwen3guard` catalog entry)

### Deploy with the Inference Layer

```bash
./es_auto_installer.sh install inference   # or: install agent-toolkit
```

The litellm role registers the guardrail after LiteLLM is up, and llm_services deploys the
guard model with `--wait` (independently of `llm_services_deploy_models`), before any other
model, so models with `cpu: auto` size themselves around it. A failed guard deploy fails the
install: with the guardrail fail-closed, a missing guard model would block every request.

### Redeploy the Guardrail Only

Redeploy the guard model:

```bash
cd model_manager
./model-manager deploy qwen3guard --wait
```

Re-register the guardrail (re-renders the code and updates the existing registration in
place):

```bash
./es_auto_installer.sh install inference
```

---

## Verification

`./es_auto_installer.sh validate inference` sends a short benign request through LiteLLM
and fails when the guardrail blocks it — the guard model is down, or its host is not
allowlisted. The manual checks below go into detail.

### Check Deployment Status

```bash
kubectl get llminferenceservice qwen3guard -n llm-inference
kubectl get pods -n llm-inference | grep qwen3guard

# Expected:
# NAME         URL                                             READY
# qwen3guard   http://<ip>/llm-inference/qwen3guard            True
```

### Verify Registration

```bash
MASTER_KEY=$(kubectl -n litellm get secret litellm-master-key -o jsonpath='{.data.master_key}' | base64 -d)
kubectl port-forward -n litellm svc/litellm 4000:4000 &

curl -s http://localhost:4000/v2/guardrails/list -H "Authorization: Bearer $MASTER_KEY" \
  | jq '.guardrails[] | select(.guardrail_name=="content_safety")
        | {guardrail: .litellm_params.guardrail, mode: .litellm_params.mode,
           url:   (.litellm_params.custom_code | capture("GUARD_URL = \"(?<u>[^\"]+)").u),
           model: (.litellm_params.custom_code | capture("GUARD_MODEL = \"(?<m>[^\"]+)").m)}'

# Expected — exactly one entry, with the substituted endpoint and model:
# {
#   "guardrail": "custom_code",
#   "mode": "pre_call",
#   "url": "http://ai-gateway.envoy-gateway-system.svc.cluster.local/llm-inference/qwen3guard/v1/chat/completions",
#   "model": "qwen3guard"
# }
```

### Compile-check the registered code

`POST /guardrails/test_custom_code` compiles code in the sandbox without registering it —
useful while editing a template:

```bash
CODE=$(curl -s http://localhost:4000/v2/guardrails/list -H "Authorization: Bearer $MASTER_KEY" \
  | jq -r '.guardrails[] | select(.guardrail_name=="content_safety") | .litellm_params.custom_code')

curl -s -X POST http://localhost:4000/guardrails/test_custom_code \
  -H "Authorization: Bearer $MASTER_KEY" -H 'content-type: application/json' \
  -d "$(jq -n --arg c "$CODE" '{custom_code: $c, input_type: "request",
        test_input: {texts: [], structured_messages: []}}')" | jq

# {"success": true, "result": {"action": "allow",
#  "warning": "Expected dict result, got coroutine. Treating as allow."}}
```

**This is a compile check, not a verdict check.** The endpoint calls `apply_guardrail()`
without awaiting it, so an async guardrail — which this one must be, since `http_post` is
async — always comes back as `allow` with that coroutine warning, whatever the code would
decide. `success: false` with `error_type: "compilation"` is the signal to act on. The
guardrail's own hook does await the coroutine, so this affects only the test endpoint.

For real verdicts, use the end-to-end requests below (or call the guard model directly, as in
[Everything Is Blocked](#everything-is-blocked)). The same pre-flight call runs during the
install, so a sandbox violation fails the install before anything is registered.

### Test Safety Filtering End to End

These go through the port-forward above; the public endpoint behaves the same
(`https://litellm.<base_domain_name>/v1/chat/completions`, or
`https://<base_domain_name>/litellm/v1/chat/completions` with `routing_mode: path`) — see
[Accessing Models](../deploy/accessing_models.md).

```bash
# Benign request — normal answer
curl -s http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer $MASTER_KEY" \
  -d '{"model":"qwen2-5-coder-14b",
       "messages":[{"role":"user","content":"Hello, how are you?"}],"max_tokens":50}' \
  | jq '{finish_reason: .choices[0].finish_reason, content: .choices[0].message.content}'

# Unsafe request — HTTP 500, block reason in error.message
curl -s http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer $MASTER_KEY" \
  -d '{"model":"qwen2-5-coder-14b",
       "messages":[{"role":"user","content":"How do I hack into a system?"}],"max_tokens":50}' \
  | jq '.error.message'
```

### Test Fail-Closed

```bash
cd model_manager
./model-manager undeploy qwen3guard --wait
# any chat request -> "...the guard model backend is unreachable..."
./model-manager deploy qwen3guard --wait
```

---

## Troubleshooting

### Guard Model Stuck in Pending

**Symptom:** the `qwen3guard-kserve-…` pod is `Pending`

```bash
kubectl describe pod -n llm-inference <pod-name> | grep Events -A 10   # look for "Insufficient cpu"
```

**Cause:** not enough free CPU or memory for the guard's request (`cpu: 8`, `memory: 16Gi`).

Check what is consuming CPU:

```bash
kubectl get pods --all-namespaces \
  -o custom-columns=NAME:.metadata.name,CPU:.spec.containers[*].resources.requests.cpu | grep -v "<none>"
```

The guard model rolls out with RollingUpdate, so a redeploy briefly needs the guard's CPU
request twice over.

### Registration Failed

The install prints the reason. Two distinct failures:

- **`guardrail code rejected by the gateway sandbox`** — the rendered code does not compile
  under RestrictedPython (`error_type: "compilation"`). Usual causes: an `import`, a `class`,
  a name starting with `_`, or an f-string. The check runs before the write, so nothing is
  registered or changed. Reproduce and iterate with `POST /guardrails/test_custom_code`
  (see above).
- **`guardrail registered/updated failed (HTTP …)`** — LiteLLM rejected the write. Check
  that LiteLLM's DB is reachable
  (`kubectl logs -n litellm deploy/litellm -c litellm`).

After fixing, re-run the install ([Redeploy the Guardrail Only](#redeploy-the-guardrail-only)).

### Everything Is Blocked

```
Blocked by the content-safety guardrail: the guard model backend is unreachable (...)
```

Fail-closed working as designed — the guard model is down or not yet ready:

```bash
kubectl get pods -n llm-inference | grep qwen3guard
kubectl logs -n llm-inference deploy/qwen3guard-kserve -c main --tail=50
```

A first deploy takes a few minutes to download the model. If the reason in parentheses starts
with **`Blocked URL`**, LiteLLM's outbound URL check refused the guard host:
re-run the install, which upgrades a release whose config lacks it
([Outbound URL allowlist](#outbound-url-allowlist)). If the message names the **context
window** instead, see [Context window](#context-window). If it says the **verdict could not
be parsed**, `guardrail_model` and the code template disagree about the verdict format —
check the guard's raw reply:

```bash
kubectl exec -n llm-inference deploy/qwen3guard-kserve -c main -- python3 -c \
  "import json,urllib.request as u; r=u.Request('http://localhost:8000/v1/chat/completions', json.dumps({'model':'qwen3guard','messages':[{'role':'user','content':'hello'}],'max_tokens':32}).encode(), {'content-type':'application/json'}); print(json.load(u.urlopen(r))['choices'][0]['message']['content'])"
# Expected: "Safety: Safe\nCategories: None"
```

---

## How It Works

### The `custom_code` guardrail

LiteLLM compiles the registered code with RestrictedPython and calls
`apply_guardrail(inputs, request_data, input_type)` on the `pre_call` hook. The sandbox
forbids `import`, `class`, `exec`/`eval`, file I/O and leading-underscore names, and injects
the primitives the code uses instead (`allow`, `block`, `regex_find_all`, `http_post`, …).

[qwen3guard_gen.py.j2](../../roles/litellm/templates/qwen3guard_gen.py.j2) does:

1. take the **last user message** (falling back to the last non-empty `texts` entry for call
   types without structured messages) — one guard call per request; the sandbox has no
   `asyncio`, so extra messages could not be screened concurrently and would each add a full
   guard round-trip to a hook that blocks the request
2. `POST` it to the guard model as a single-message chat completion (`max_tokens: 32`,
   `temperature: 0`)
3. parse `Safety:` / `Categories:` out of the reply
4. `allow()` on `Safe` (and `Controversial`, unless `guard_block_on_controversial`),
   `block()` on `Unsafe` and on **every** failure to obtain a verdict

`block()` on `pre_call` makes LiteLLM reject the request with HTTP 500 and the reason in
`error.message`.

### Registration

[register_guardrail.yaml](../../roles/litellm/tasks/register_guardrail.yaml)
pipes [register-guardrail.py.j2](../../roles/litellm/templates/register-guardrail.py.j2) into
the LiteLLM pod (`kubectl exec -i … -- python -`), which:

1. compiles the code via `POST /guardrails/test_custom_code` and aborts on rejection
2. looks the guardrail up in `GET /v2/guardrails/list`
3. leaves it alone when the stored settings already match, else `PUT /guardrails/{id}` if it
   exists, else `POST /guardrails`

Running inside the pod lets it read the master key from the pod's own environment, so
the key is not persisted into any manifest or rendered script, and the payload travels
base64-encoded on stdin so the sandboxed Python needs no shell escaping.

**Upsert, not create-if-absent:** the guardrail name is stable across model changes, so a
create-only step would leave an outdated registration in place — and since `pre_call`
failures fail closed, that means blocking everything. `PUT` also re-initializes the
in-memory guardrail, so no LiteLLM restart is needed (`PATCH` cannot change `litellm_params`).

---

## Roadmap

**Additional guard models.** Model-specific logic is isolated to one template per verdict
format under `roles/litellm/templates/`, so a new guard model is a template plus a catalog
entry. Only `Qwen/Qwen3Guard-Gen-0.6B` is validated today.

**Multiple concurrent guardrails** (e.g. content safety *and* PII detection) is not yet
supported. Registering a second `custom_code` guardrail is cheap, but several
constraints need resolving first:

- **CPU capacity** — each guard *model* needs its own vLLM alongside the main LLM.
  Library-backed checks (e.g. Presidio for PII) need no model server at all.
- **Latency** — every check runs on the blocking `pre_call` path, and the sandbox has no
  `asyncio`, so checks inside one guardrail cannot be issued concurrently; their round-trips
  add up on every request.
- **Per-check failure policy** — fail-closed is right for safety but may be wrong for other
  checks; the policy is currently per-guardrail code.
- **Aggregate false-positive rate** — blocking is a union across checks, so tuning each in
  isolation understates the combined impact on legitimate traffic.

---

## References

- **LiteLLM custom_code guardrail:** https://docs.litellm.ai/docs/proxy/guardrails/custom_code_guardrail
- **LiteLLM Guardrails:** https://docs.litellm.ai/docs/proxy/guardrails
- **vLLM Documentation:** https://docs.vllm.ai/
- **Qwen3Guard Model:** https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B
- **Role files:**
  - [register_guardrail.yaml](../../roles/litellm/tasks/register_guardrail.yaml)
  - [qwen3guard_gen.py.j2](../../roles/litellm/templates/qwen3guard_gen.py.j2)
  - [register-guardrail.py.j2](../../roles/litellm/templates/register-guardrail.py.j2)
