# Troubleshooting

Symptoms you are most likely to hit, and what to do about them.

## Deploy and storage

| Symptom | Likely cause / fix |
|---------|--------------------|
| `PVC 'model-store' ... not ready (Pending)` | Normal on `WaitForFirstConsumer` storage — it binds once the pod schedules. A `Pending` PVC on `Immediate` storage means a provisioner problem. |
| Download Job `Permission denied` | The downloader runs non-root; ensure the PVC is writable. The Job sets `HOME` and the cache paths to writable locations. |
| `ImagePullBackOff` on serving pods | Registry rate limit or proxy issue — configure a pull-through mirror, or pre-pull the runtime image on the node. |
| Model "gated" error | Set `HF_TOKEN` and re-run `./model-manager deploy <name> --wait`. See [Gated models](deploy/deploy_models.md#gated-models-hugging-face-token). |
| Model stuck not-Ready | `kubectl logs -n llm-inference -l app.kubernetes.io/name=<name> -f`. CPU model load plus warmup takes minutes — this is often just slow, not broken. |
| Download fails behind a corporate proxy | Set `http_proxy` / `https_proxy` / `no_proxy` in the catalog's `network:` section, or via `MM_HTTP_PROXY` / `MM_HTTPS_PROXY` / `MM_NO_PROXY`. |
| `ovms --pull` aborts: *"unsupported for OpenVINO models"* | `weight_format` was set for a model already published as OpenVINO IR. Leave it unset for `OpenVINO/*` repos — see [Runtimes](customize/runtimes.md#weight_format--only-for-raw-hugging-face-repos). |

## CPU pinning and scheduling

| Symptom | Likely cause / fix |
|---------|--------------------|
| Deploy blocked by the NRI pre-flight capacity check | The requested `cpu` × `replicas` doesn't fit the balloon pool. Lower the request, or override with `MM_NRI_PREFLIGHT=warn` / `MM_NRI_SKIP_PREFLIGHT=1`. |
| Re-running `deploy` or `install` aborts on pre-flight for an already-deployed model | Expected before the spec-diff no-op landed. An unchanged model is now skipped without re-running pre-flight. Use `MM_FORCE_REDEPLOY=1` to force a full re-apply. |
| Pod rejected with `SMTAlignmentError` | Under `kubelet-static` with `full-pcpus-only`, `cpu` must be a multiple of the SMT width. On an SMT-disabled cluster set `mm_kubelet_threads_per_core: 1`. |
| Pod annotated `cpu-policy.model-manager.io/reason: nri-resolution-failed` | NRI was expected but couldn't be resolved, so the pod fell back to `best-effort`. It still serves, without exclusive cores. Check the NRI plugin is running on the node. |
| Model pod stays `Pending` with no node available | Likely the `require_amx: true` affinity with no AMX-labelled node. Check NFD is running: `kubectl get pods -n node-feature-discovery`, then `kubectl get nodes -L feature.node.kubernetes.io/cpu-security.amx.enabled`. |

Detail → [CPU Pinning & NUMA](reference/cpu_pinning.md).

## Requests and endpoints

| Symptom | Likely cause / fix |
|---------|--------------------|
| `404` from `inference.<domain>` in `litellm_vkey` mode | Nothing is bound to that host in this mode — use `litellm.<domain>`. See [Accessing Models](deploy/accessing_models.md). |
| `401` on every request in `jwt` mode | Missing, malformed, or expired JWT. Tokens last 15 minutes by default — re-run `source .../get-keycloak-token.sh`, or pass `--lifespan 3600`. |
| `401` in `litellm_vkey` mode with a key that used to work | The virtual key expired, was revoked, or exceeded its budget. List keys with `/key/list?return_full_object=true`. |
| `401` for one model only, in `litellm_vkey` mode | The key's model allow-list doesn't include it. Mint a key with that model in `models`. |
| TLS certificate warnings from `curl` or a browser | Self-signed CA. Use `--cacert env/<env>/logs/ai-solutions-ca.crt`, or import that CA into your trust store. |
| The LiteLLM UI redirects to `http://` and fails | The trailing slash is required: `https://litellm.<domain>/ui/`, not `/ui`. |
| Model deployed, but requests return "model not found" | The `model` field must match the deploy **name**, not the Hugging Face `model_id`. List what's registered with `GET /v1/models`. |

## Install

| Symptom | Likely cause / fix |
|---------|--------------------|
| No models deployed after `install inference` | Expected. `config.yaml` ships `llm_services_deploy_models: false`, and only models flagged `autodeploy: true` deploy. **Both** are required — see [Auto-deploy on install](customize/catalog.md#auto-deploy-on-install). |
| An auto-deployed model failed but the install reported success | By design — the serving infrastructure is up and the failure is in the summary. Set `model_deploy_strict: true` to make it fatal. |
| `--server-version` rejected for an OpenVINO model | OpenVINO is a flat runtime with no version axis. Only vLLM is versioned. |
| `--server-version` rejected for a vLLM model | The catalog's `servers:` block is an allow-list. Add the version there first — see [Model Catalog](customize/catalog.md#choose-which-server-and-version-a-model-may-use). |

## Getting more detail

```bash
# What model-manager would do, without doing it
./model-manager deploy <name> --dry-run

# Serving objects and their conditions
kubectl get llminferenceservices,inferenceservices -n llm-inference
kubectl describe llminferenceservice <name> -n llm-inference

# Model logs
kubectl logs -n llm-inference -l app.kubernetes.io/name=<name> -f

# The download Job for a model
kubectl get jobs -n llm-inference
kubectl logs -n llm-inference job/<download-job-name>

# Gateway and route state
kubectl get gateway,httproute -A
kubectl get aigatewayroute -A          # ai_gateway_provider=envoy
```

For platform-level problems — cluster, storage, certificates, DNS — see the
[solutions documentation](https://github.com/intel/enterprise-ai-solutions/blob/main/docs/README.md).
