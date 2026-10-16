# Deploy on OpenShift

> [!WARNING]
> **Initial release, work in progress.** Validated on one single-node OpenShift
> 4.22 cluster with OpenShift AI 3.3. Expect gaps (see [Known gaps](#known-gaps)).

On OpenShift the inference layer does not install KServe, the AI gateways or NFD.
OpenShift AI and the marketplace operators provide them. `install inference` checks
that they are there, publishes OpenShift AI's inference Gateway behind Keycloak, and
pins model pods to CPU balloons. `model-manager` then renders models for OpenShift AI.

Everything is driven by `cluster_distribution`, which the platform's preflight
detects. Nothing in `global_config.yaml` or `models.yaml` changes between platforms.

## Before you start

1. **Install the prerequisites:** the operators, the DataScienceCluster and a default
   StorageClass. Follow
   [Deploy on OpenShift](https://github.com/intel/enterprise-ai-solutions/blob/main/docs/deploy/openshift.md)
   in the platform repo.
2. **Install the platform layer:** run `./es_auto_installer.sh install platform --env <env>`
   (see that guide's *Installer configuration*).

## Install the inference layer

```bash
./es_auto_installer.sh init inference --env ocp     # once
./es_auto_installer.sh install inference --env ocp
```

| Component | On OpenShift |
|-----------|--------------|
| `kserve` | **Checks** that OpenShift AI is installed: a DataScienceCluster with KServe `Managed`, the KServe controllers in `redhat-ods-applications`, and the LLMInferenceService and InferencePool CRDs. It installs nothing. Then it sets up the inference Gateway `openshift-ingress/openshift-ai-inference`: it creates the Gateway if it is missing, copies the gateway TLS certificate, publishes it through a passthrough Route on `inference.<domain>`, and adds a Gateway-wide Keycloak `AuthPolicy` (Connectivity Link) when `auth_provider: keycloak`. Teardown does nothing. |
| `nfd` | **Checks** that the NFD operator runs a NodeFeatureDiscovery instance, and lists the nodes that have AMX. It installs nothing, and teardown does nothing. |
| `envoy_ai_gateway`, `agentgateway` | Skipped. Models are routed by OpenShift AI's inference Gateway at `/<namespace>/<model>/v1/...`. There is no unified `/v1` endpoint (AIGatewayRoute). |
| `keycloak_config` | Unchanged, except that `kcadm` keeps its session in `/tmp`, because RHBK's `HOME=/` is not writable for the UID that the SCC assigns. |
| `nri_cpu_balloons` (platform repo) | Grants the SCC, and selects OpenShift AI model pods into `vllm-balloon-tp1` by their `kserve.io/component=workload` label. v1alpha1 cannot annotate pods (see below). |
| `llm_services` | Waits for the KServe controller in `redhat-ods-applications`. Gives the model-store PVC `ReadWriteOnce` access when the StorageClass is RWO-only (`storage_custom_rwx: false`, e.g. LVMS). Passes `MM_PLATFORM` and the inference Gateway to `model-manager`. |

## Deploy a model

`model-manager` detects the platform on every run, before it renders anything. So the
commands are the same as in [Deploy a Model](deploy_models.md):

```bash
cd model_manager
export KUBECONFIG=~/.kube/config
export MM_CONFIG=../env/ocp/models.yaml              # or wherever your catalog lives
export MM_GLOBAL_CONFIG=../env/ocp/global_config.yaml # for the inference hostname

./model-manager deploy qwen3-0-6b --dry-run          # look at what is rendered
./model-manager deploy qwen3-0-6b --wait
```

What changes on OpenShift:

| | Kubernetes | OpenShift |
|---|---|---|
| Platform | `MM_PLATFORM=kubernetes` | `MM_PLATFORM=openshift`. Detected from the `config.openshift.io` ClusterVersion API, or set explicitly. |
| LLMInferenceService API | `v1alpha2` | The newest version the cluster serves: OpenShift AI 3.3 serves only `v1alpha1` (`MM_LLMISVC_API_VERSION`). |
| Gateway | `eg-gateway` plus an AIGatewayRoute | `openshift-ingress/openshift-ai-inference` (`MM_INFERENCE_GATEWAY`, `MM_INFERENCE_GATEWAY_NS`). `MM_AIGATEWAY_ENABLED=false`. |
| vLLM flags | Container `args` | `VLLM_ADDITIONAL_ARGS`, because OpenShift AI's template starts vLLM itself. `--served-model-name` becomes `spec.model.name`. The probes use HTTPS (vLLM serves TLS). |
| Pod UIDs | Pinned `runAsUser` / `runAsGroup` / `fsGroup` | Dropped, because the `restricted-v2` SCC assigns them from the namespace's range. |
| CPU balloon | Pod annotation | The `kserve.io/component=workload` label. v1alpha1 puts the policy's annotations on the LLMInferenceService itself, not on its pods. |
| InferencePool status | Patched by `model-manager` | Reported by the Gateway controller itself. |

Check the deployment:

```bash
oc get llminferenceservice -n llm-inference      # READY True, URL on openshift-ai-inference
oc exec -n llm-inference deploy/qwen3-0-6b-kserve -c main -- \
  grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/1/status   # the balloon's cpuset
```

## Call the model

Models are served at `https://inference.<domain>/<namespace>/<model>/v1/...`, through
the OpenShift router, and need a Keycloak token. If the `*.apps` hostnames don't resolve
from your machine, give the ingress VIP with `--resolve`, or with `MM_GATEWAY_IP` for
the token script.

```bash
# The token script reads the client secret from the cluster.
MM_GATEWAY_IP=<ingress VIP> source model_manager/scripts/get-keycloak-token.sh

H=inference.<domain>
curl -s --noproxy '*' --cacert env/ocp/logs/ai-solutions-ca.crt \
  --resolve "$H:443:<ingress VIP>" \
  https://$H/llm-inference/qwen3-0-6b/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-0-6b","messages":[{"role":"user","content":"Hello /no_think"}],"max_tokens":64}'
```

Without the script:

```bash
SECRET=$(oc get secret keycloak-client-secret -n keycloak -o jsonpath='{.data.client-secret}' | base64 -d)
TOKEN=$(curl -s --noproxy '*' --cacert env/ocp/logs/ai-solutions-ca.crt \
  --resolve "keycloak.<domain>:443:<ingress VIP>" \
  https://keycloak.<domain>/realms/inference/protocol/openid-connect/token \
  -d grant_type=client_credentials -d client_id=inference-client -d client_secret=$SECRET | jq -r .access_token)
```

Response codes: no token or an invalid one gives `401`, and a valid one `200`.

## Known gaps

| Gap | Today |
|-----|-------|
| OpenVINO models (`InferenceService`) | Not validated. They need a `ClusterServingRuntime` that OpenShift AI does not ship. |
| TP2+ balloons | On v1alpha1 every model pod matches by label into `vllm-balloon-tp1`, so models with more than one NUMA node per replica don't get their own balloon type. |
| Unified `/v1` endpoint | Not available, because there is no AI gateway. Clients (e.g. eRAG) need the per-model URL. |
| Token script fallback | When the hostnames neither resolve nor `MM_GATEWAY_IP` is set, the port-forward fallback fails: RHBK serves plain HTTP on `8080`. |
| NRI plugin sync | The installer restarts the plugin once it applies the node policy. After a balloon policy change of your own, restart it by hand, or new pods may run unpinned: `oc delete pod -n kube-system -l app.kubernetes.io/name=nri-resource-policy-balloons`. |

## Related Docs

- [Deploy a Model](deploy_models.md): every `model-manager` command
- [Accessing Models](accessing_models.md): endpoints and credentials on Kubernetes
- [CPU Pinning & NUMA](../reference/cpu_pinning.md): the balloons policy
- [Deploy on OpenShift (platform)](https://github.com/intel/enterprise-ai-solutions/blob/main/docs/deploy/openshift.md): prerequisites and the platform layer
