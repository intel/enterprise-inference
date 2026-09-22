#!/usr/bin/env bash
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Core helpers — logging, template rendering, kubectl wrappers

# ── Colors ──
if [[ -t 1 ]]; then
    BOLD=$'\033[1m' RED=$'\033[1;31m' GREEN=$'\033[1;32m'
    YELLOW=$'\033[1;33m' BLUE=$'\033[1;34m' CYAN=$'\033[1;36m'
    DIM=$'\033[90m' RESET=$'\033[0m'
else
    BOLD='' RED='' GREEN='' YELLOW='' BLUE='' CYAN='' DIM='' RESET=''
fi

info()  { printf "${BLUE}[INFO ]${RESET}  %s\n" "$*"; }
ok()    { printf "${GREEN}[  OK ]${RESET}  %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN ]${RESET}  %s\n" "$*" >&2; }
err()   { printf "${RED}[ERROR]${RESET}  %s\n" "$*" >&2; }
abort()   { err "$@"; exit 1; }

check_prerequisites() {
    local missing=()
    for tool in kubectl yq jq curl; do
        command -v "$tool" &>/dev/null || missing+=("$tool")
    done
    [[ ${#missing[@]} -eq 0 ]] && return 0

    info "Missing tools: ${missing[*]} — attempting install..."
    local pkg_mgr=""
    if command -v apt-get &>/dev/null; then pkg_mgr="apt"
    elif command -v dnf &>/dev/null; then pkg_mgr="dnf"
    elif command -v yum &>/dev/null; then pkg_mgr="yum"
    fi

    for tool in "${missing[@]}"; do
        case "$tool" in
            yq)
                local yq_version="v4.44.6"
                local yq_url="https://github.com/mikefarah/yq/releases/download/${yq_version}/yq_linux_amd64"
                info "Installing yq ${yq_version} from GitHub..."
                sudo curl -sSL "$yq_url" -o /usr/local/bin/yq && sudo chmod +x /usr/local/bin/yq \
                    || abort "Failed to install yq. Install manually: sudo curl -sSL $yq_url -o /usr/local/bin/yq && sudo chmod +x /usr/local/bin/yq"
                ;;
            jq|curl)
                [[ -z "$pkg_mgr" ]] && abort "Cannot auto-install $tool — no supported package manager found. Install manually."
                info "Installing $tool via $pkg_mgr..."
                case "$pkg_mgr" in
                    apt) sudo apt-get install -y -qq "$tool" 2>/dev/null || abort "Failed to install $tool via apt" ;;
                    dnf) sudo dnf install -y -q "$tool" 2>/dev/null || abort "Failed to install $tool via dnf" ;;
                    yum) sudo yum install -y -q "$tool" 2>/dev/null || abort "Failed to install $tool via yum" ;;
                esac
                ;;
            kubectl)
                abort "kubectl not found. Install it: https://kubernetes.io/docs/tasks/tools/ — then export KUBECONFIG=<path-to-kubeconfig> to connect to your cluster."
                ;;
        esac
    done

    for tool in kubectl yq jq curl; do
        command -v "$tool" &>/dev/null || abort "Required tool still not found after install attempt: $tool"
    done
    ok "All prerequisites installed"
}

# Verify kubectl can reach the API server. Fails with actionable guidance.
ensure_cluster_reachable() {
    if kubectl get --raw='/readyz' --request-timeout=5s >/dev/null 2>&1; then
        return 0
    fi

    err "kubectl cannot reach the Kubernetes API server."
    if [[ -n "${KUBECONFIG:-}" ]]; then
        err "  Current KUBECONFIG: $KUBECONFIG"
    else
        err "  KUBECONFIG is not set and \$HOME/.kube/config does not exist."
    fi
    err "  Export a valid kubeconfig, e.g.:"
    err "    export KUBECONFIG=<repo-root>/env/local/kubeconfig.yaml"
    err "  Then verify:  kubectl get nodes"
    abort "kubectl is not connected to a cluster"
}

# ── Template rendering ──

render_template() {
    local template="$1"; shift
    [[ -f "$template" ]] || abort "Template not found: $template"
    local content=$(<"$template")
    local pair
    for pair in "$@"; do
        local key="${pair%%=*}" value="${pair#*=}"
        content="${content//\{\{${key}\}\}/$value}"
    done
    echo "$content"
}

# Emit an arbitrary string as a safely-quoted YAML scalar. Needed for values we
# do not control the shape of — notably `chat_template`, whose Jinja routinely
# contains double quotes, backslashes and newlines. Naively wrapping such a value
# in "..." produced a manifest that either failed to parse or silently changed
# meaning. jq's @json emits a JSON string, which is valid YAML flow scalar syntax
# and handles the escaping for us.
yaml_quote() { jq -rn --arg v "$1" '$v | @json'; }

# ── Kubernetes helpers ──

# apply_chat_template_cm — store an inline Jinja chat template in a ConfigMap.
#
# Inline Jinja CANNOT be passed to vLLM through the LLMInferenceService spec: the
# KServe llmisvc controller marshals the whole spec and runs it through Go's
# text/template (pkg/controller/v1alpha2/llmisvc/config_merge.go), so a template
# containing `{{ ... }}` is intercepted before vLLM ever sees it and the CR fails
# to reconcile with e.g.
#     failed to parse template config: template: config:1: bad character U+005B '['
# (U+005B is the `[` of `message['content']`). Reconciliation then fails silently
# — generation/observedGeneration still match, but the child Deployment keeps the
# previous args. Empirically `{% if %}` survives; any `{{ }}` does not, so every
# real chat template is affected.
#
# So the Jinja is kept OUT of the spec entirely: it goes in a ConfigMap, the
# ConfigMap is mounted into the serving pod, and the spec carries only a file
# path — which is inert to Go templating.
#
# The manifest is built with jq (not a heredoc) so the template is transported as
# a JSON string. That escapes arbitrary content correctly and — unlike a YAML `|`
# block scalar, which the YAML spec requires to normalize line breaks — preserves
# literal CR, so templates like falcon-7b-instruct's `.replace('\r\n','\n')`
# survive byte-exact.
apply_chat_template_cm() {
    local name="$1" namespace="$2" template="$3"
    jq -n --arg n "$name" --arg ns "$namespace" --arg v "$template" \
        '{apiVersion:"v1", kind:"ConfigMap",
          metadata:{name:$n, namespace:$ns,
                    labels:{"app.kubernetes.io/managed-by":"model-manager"}},
          data:{"chat_template.jinja":$v}}' \
      | kubectl apply -f - >/dev/null
}

kube_apply() { kubectl apply -f - <<< "$1"; }
kube_ssa()   { kubectl apply --server-side --field-manager=model-manager --force-conflicts -f - <<< "$1"; }

ensure_namespace() { kubectl get namespace "$1" &>/dev/null || kubectl create namespace "$1"; }

crd_ready() {
    local kind="$1" namespace="$2" name="$3"
    local raw
    raw=$(kubectl get "$kind" "$name" -n "$namespace" -o json 2>/dev/null) || { echo "NotFound"; return 0; }
    local result
    result=$(echo "$raw" | jq -r '.status.conditions[]? | select(.type=="Ready") | .status // empty' 2>/dev/null) || true
    [[ -z "$result" ]] && { echo "Pending"; return 0; }
    echo "$result"
}

# workload_ready — True once the model's serving workload is up and passing
# probes, INDEPENDENT of gateway/InferencePool wiring.
#
# For an LLMInferenceService the overall `Ready` condition also requires the
# InferencePool to be Accepted — but that acceptance is something model-manager
# itself must patch (patch_pool_status). Gating deploy readiness on `Ready`
# therefore deadlocks: we'd wait for a condition that only WE can satisfy.
# So we key on MainWorkloadReady (the vLLM pod is loaded + serving) and let the
# ready-handler perform the pool patch. Falls back to `Ready` for other kinds
# (e.g. plain InferenceService) that don't expose MainWorkloadReady.
workload_ready() {
    local kind="$1" namespace="$2" name="$3"
    local raw
    raw=$(kubectl get "$kind" "$name" -n "$namespace" -o json 2>/dev/null) || { echo "NotFound"; return 0; }
    local main
    main=$(echo "$raw" | jq -r '.status.conditions[]? | select(.type=="MainWorkloadReady") | .status // empty' 2>/dev/null) || true
    [[ -n "$main" ]] && { echo "$main"; return 0; }
    local rdy
    rdy=$(echo "$raw" | jq -r '.status.conditions[]? | select(.type=="Ready") | .status // empty' 2>/dev/null) || true
    [[ -z "$rdy" ]] && { echo "Pending"; return 0; }
    echo "$rdy"
}

crd_delete()        { kubectl delete "$1" "$3" -n "$2" --ignore-not-found 2>/dev/null || true; }
pvc_status()        { kubectl get pvc "$2" -n "$1" -o jsonpath='{.status.phase}' 2>/dev/null || true; }
runtime_installed() { kubectl get clusterservingruntime "$1" &>/dev/null; }

gateway_ip() {
    local ns="${MM_EG_GATEWAY_NS:-envoy-gateway-system}"
    local gw="${MM_EG_GATEWAY_NAME:-eg-gateway}"
    kubectl get svc -n "$ns" \
        -l "gateway.envoyproxy.io/owning-gateway-name=$gw" \
        -o jsonpath='{.items[0].status.loadBalancer.ingress[0].ip}' 2>/dev/null || true
}

wait_job() {
    kubectl wait --for=condition=complete "job/$2" -n "$1" --timeout="${3:-7200}s" 2>/dev/null
}

stream_job_logs() {
    local pod
    pod=$(kubectl get pods -n "$1" -l "job-name=$2" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    [[ -n "$pod" ]] && kubectl logs -f "$pod" -n "$1" 2>/dev/null
}

# fmt_duration — seconds → compact "Nm Ns" (or "Ns" under a minute).
fmt_duration() {
    local s="$1"
    (( s < 60 )) && { echo "${s}s"; return; }
    echo "$((s / 60))m $((s % 60))s"
}

# pod_progress — one-line human status for a model's serving pod, used by the
# deploy --wait heartbeat. Reports pod phase plus a hint of what it's doing
# (pulling image, waiting to schedule, loading/warming up, or a failure reason).
pod_progress() {
    local ns="$1" model="$2"
    local json
    json=$(kubectl get pods -n "$ns" -l "app.kubernetes.io/name=$model" \
        -o jsonpath='{.items[0].status.phase}{"|"}{.items[0].status.containerStatuses[0].ready}{"|"}{.items[0].status.containerStatuses[0].state.waiting.reason}' 2>/dev/null) || json=""

    [[ -z "$json" || "$json" == "|" ]] && { echo "waiting for pod to be created"; return; }

    local phase ready wait_reason
    IFS='|' read -r phase ready wait_reason <<< "$json"

    case "$phase" in
        Pending)
            case "$wait_reason" in
                ContainerCreating) echo "creating container (pulling image / mounting weights)" ;;
                ImagePullBackOff|ErrImagePull) echo "image pull failed ($wait_reason)" ;;
                CrashLoopBackOff) echo "container crashing ($wait_reason) — check logs" ;;
                "") echo "pending (waiting to be scheduled)" ;;
                *) echo "pending ($wait_reason)" ;;
            esac ;;
        Running)
            [[ "$ready" == "true" ]] && echo "running — finalizing routes" \
                                     || echo "running — loading model / warming up" ;;
        Failed)    echo "pod failed — check logs" ;;
        Succeeded) echo "pod exited" ;;
        *)         echo "${phase:-unknown}" ;;
    esac
}

# wait_pods_gone — block until no pods for a model remain in the namespace.
#
# Deleting the serving CR (LLMInferenceService/InferenceService) returns as soon
# as the CR object itself is gone, but the KServe-owned Deployment → ReplicaSet →
# Pods are torn down asynchronously by the garbage collector and honour their
# graceful-termination period. Those pods stay in phase Running (Terminating)
# and keep their pinned CPUs / NRI balloon until they are truly removed — which
# is why an immediate cpu-collisions-check still lists them. This waits for that
# tail so `undeploy --wait` only returns once the CPUs are actually reclaimed.
#
# Bounded by the timeout so a pod wedged by a finalizer or stuck termination
# never blocks the CLI forever. Returns 0 once the pods are gone, 1 on timeout.
wait_pods_gone() {
    local namespace="$1" name="$2" timeout="${3:-900}"
    local selector="app.kubernetes.io/name=$name"
    local start=$SECONDS end=$((SECONDS + timeout))
    local last_beat=0 remaining
    while (( SECONDS < end )); do
        remaining=$(kubectl get pods -n "$namespace" -l "$selector" \
            -o name 2>/dev/null | wc -l)
        remaining=${remaining//[[:space:]]/}
        (( remaining == 0 )) && return 0
        if (( SECONDS - last_beat >= 15 )); then
            info "  ${DIM}[$(fmt_duration $((SECONDS - start)))] waiting for ${remaining} pod(s) to terminate...${RESET}"
            last_beat=$SECONDS
        fi
        sleep 3
    done
    return 1
}


patch_pool_status() {
    local name="$1" namespace="$2"
    local pool="${name}-inference-pool"

    local i
    for i in $(seq 1 24); do
        kubectl get inferencepool "$pool" -n "$namespace" &>/dev/null && break
        sleep 5
    done
    kubectl get inferencepool "$pool" -n "$namespace" &>/dev/null || return 0

    # The InferencePool is parented to the AI Gateway (inference-pool-with-aigwroute class),
    # NOT to the public eg-gateway. Read the actual gateway ref from the LLMInferenceService
    # spec; fall back to MM_INFERENCE_GATEWAY / ai-gateway defaults.
    local gw_name gw_ns
    gw_name=$(kubectl get llminferenceservice "$name" -n "$namespace" \
        -o jsonpath='{.spec.router.gateway.refs[0].name}' 2>/dev/null) || true
    gw_ns=$(kubectl get llminferenceservice "$name" -n "$namespace" \
        -o jsonpath='{.spec.router.gateway.refs[0].namespace}' 2>/dev/null) || true
    gw_name="${gw_name:-${MM_INFERENCE_GATEWAY:-ai-gateway}}"
    gw_ns="${gw_ns:-${MM_INFERENCE_GATEWAY_NS:-llm-inference}}"

    # The controllerName in the parent status must match the GatewayClass's
    # controller for the acceptance to be attributed to the right gateway.
    # ai_gateway_provider() is defined in routing.sh (sourced alongside this file).
    local controller="gateway.envoyproxy.io/gatewayclass-controller"
    if [[ "$(ai_gateway_provider 2>/dev/null)" == "agentgateway" ]]; then
        controller="agentgateway.dev/agentgateway"
    fi

    local now; now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    kubectl patch inferencepool "$pool" -n "$namespace" --type=merge --subresource=status -p \
        "{\"status\":{\"parents\":[{\"parentRef\":{\"group\":\"gateway.networking.k8s.io\",\"kind\":\"Gateway\",\"name\":\"${gw_name}\",\"namespace\":\"${gw_ns}\"},\"controllerName\":\"${controller}\",\"conditions\":[{\"type\":\"Accepted\",\"status\":\"True\",\"reason\":\"Accepted\",\"message\":\"Patched by model-manager\",\"lastTransitionTime\":\"${now}\"}]}]}}" \
        2>/dev/null || true
}

ensure_storage() {
    local namespace="$1"
    local pvc_name="${STORAGE_PVC_NAME:-model-store}"

    ensure_namespace "$namespace"

    local pvc_phase
    pvc_phase=$(pvc_status "$namespace" "$pvc_name")

    if [[ -z "$pvc_phase" ]]; then
        abort "PVC '$pvc_name' not found in namespace '$namespace'. Provision storage with the infrastructure playbook before deploying models."
    elif [[ "$pvc_phase" == "Bound" ]]; then
        ok "Storage ready: $pvc_name ($pvc_phase)"
    elif [[ "$pvc_phase" == "Pending" ]]; then
        # A Pending PVC is expected — not an error — when its StorageClass uses
        # WaitForFirstConsumer binding (e.g. rancher.io/local-path): the volume
        # only binds once a consuming pod is scheduled, which is the very pod
        # this deploy is about to create. Aborting here would deadlock. Only a
        # Pending PVC on an Immediate-binding StorageClass is a real fault.
        local sc binding_mode
        sc=$(kubectl get pvc "$pvc_name" -n "$namespace" -o jsonpath='{.spec.storageClassName}' 2>/dev/null || true)
        binding_mode=$(kubectl get storageclass "$sc" -o jsonpath='{.volumeBindingMode}' 2>/dev/null || true)
        if [[ "$binding_mode" == "WaitForFirstConsumer" ]]; then
            ok "Storage ready: $pvc_name (Pending — binds on first consumer, StorageClass '$sc' uses WaitForFirstConsumer)"
        else
            abort "PVC '$pvc_name' in namespace '$namespace' is not ready (phase: Pending, StorageClass '${sc:-?}' binding '${binding_mode:-Immediate}'). Check your storage provisioner."
        fi
    else
        abort "PVC '$pvc_name' in namespace '$namespace' is not ready (phase: $pvc_phase). Wait for it to become Bound or check your storage provisioner."
    fi
}
