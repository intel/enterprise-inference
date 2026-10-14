#!/usr/bin/env bash
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Routing — LiteLLM registration + AI Gateway route management
#
# Auth-mode detection (precedence):
#   1. MM_LITELLM_ENABLED=true env var → litellm mode
#   2. auth_type → jwt | litellm_vkey (or deprecated auth_provider: keycloak | litellm),
#      global_config.yaml first then config.inference.yaml, as the installer loads them
#   3. Default → jwt
#
# Both modes route through the same AI Gateway unified URL.

# ── Auth / Provider detection ──

auth_type() {
    # Env override takes precedence
    local val="${MM_LITELLM_ENABLED:-}"
    [[ "$val" =~ ^(true|1|yes)$ ]] && { echo "litellm_vkey"; return; }
    [[ "$val" =~ ^(false|0|no)$ ]] && { echo "jwt"; return; }

    local cfg provider
    for cfg in "$(discover_global_config 2>/dev/null)" "$(discover_inference_config 2>/dev/null)"; do
        [[ -n "$cfg" && -f "$cfg" ]] || continue
        provider=$(yq e '.auth_type // .auth_provider // ""' "$cfg" 2>/dev/null)
        [[ "$provider" =~ ^(litellm_vkey|litellm)$ ]] && { echo "litellm_vkey"; return; }
        [[ "$provider" =~ ^(jwt|keycloak)$ ]] && { echo "jwt"; return; }
    done

    echo "jwt"
}

# ── LiteLLM ──

LITELLM_NAMESPACE="${MM_LITELLM_NAMESPACE:-litellm}"
LITELLM_SERVICE="${MM_LITELLM_SERVICE:-litellm}"
LITELLM_SECRET="${MM_LITELLM_SECRET:-litellm-master-key}"

litellm_enabled() {
    [[ "$(auth_type)" == "litellm_vkey" ]]
}

litellm_url() {
    [[ -n "${MM_LITELLM_URL:-}" ]] && { echo "${MM_LITELLM_URL%/}"; return; }
    local port
    port=$(kubectl get svc "$LITELLM_SERVICE" -n "$LITELLM_NAMESPACE" \
        -o jsonpath='{.spec.ports[0].port}' 2>/dev/null) || true
    [[ -n "$port" ]] && echo "http://$LITELLM_SERVICE.$LITELLM_NAMESPACE.svc.cluster.local:$port"
    return 0
}

litellm_key() {
    [[ -n "${MM_LITELLM_MASTER_KEY:-}" ]] && { echo "$MM_LITELLM_MASTER_KEY"; return; }
    local encoded
    encoded=$(kubectl get secret "$LITELLM_SECRET" -n "$LITELLM_NAMESPACE" \
        -o jsonpath='{.data.master_key}' 2>/dev/null) || true
    [[ -n "$encoded" ]] && echo "$encoded" | base64 -d
    return 0
}

litellm_curl() {
    local method="$1" url="$2" data="${3:-}"
    local key; key=$(litellm_key)
    local args="-sf -X $method -H 'Content-Type: application/json'"
    [[ -n "$key" ]] && args+=" -H 'Authorization: Bearer $key'"
    [[ -n "$data" ]] && args+=" -d '$data'"
    args+=" '$url'"
    kubectl run "mm-curl-${RANDOM:-$$}" --rm -i --restart=Never \
        --image="${MM_CURL_IMAGE:-curlimages/curl:8.12.1}" -n "$LITELLM_NAMESPACE" \
        -- sh -c "curl $args" 2>/dev/null
}

# Resolve every UUID LiteLLM has registered under $model_name.
#
# /model/info is the only way back from a human model_name to the UUIDs LiteLLM
# assigned it. Shared by deregister (delete each) and register (confirm the new
# entry actually landed).
#
# litellm_curl output can have trailing noise appended by `kubectl run`
# (e.g. 'pod "mm-curl-x" deleted'). Two consequences under `set -euo pipefail`:
#   1. jq errors on the trailing garbage and exits non-zero, which would abort
#      the whole script — hence the trailing `|| true` on the pipe.
#   2. so we strip everything after the last '}' first, leaving clean JSON.
litellm_model_ids() {
    local model_name="$1" url="$2"
    local info
    info=$(litellm_curl GET "$url/model/info") || return 0
    info="${info%\}*}}"   # trim trailing 'pod "..." deleted' after the final '}'
    printf '%s' "$info" | jq -rc --arg n "$model_name" \
        '.data[]? | select(.model_name == $n) | .model_info.id' 2>/dev/null || true
}

litellm_register() {
    local model_name="$1" model_url="$2"
    litellm_enabled || return 0

    local litellm_base; litellm_base=$(litellm_url)
    [[ -z "$litellm_base" ]] && { warn "LiteLLM not reachable — skipping registration"; return 0; }

    # Make this idempotent. /model/new mints a FRESH UUID on every call and does
    # not upsert on model_name, so a redeploy — or a retry below — silently
    # accumulates duplicate entries pointing at possibly-stale api_base values,
    # and the router load-balances across them. Clear the name first.
    litellm_deregister "$model_name"

    # LiteLLM's OpenAI provider appends /chat/completions (or /embeddings) to api_base.
    # Caller passes the version-prefixed root (e.g. .../v1) so LiteLLM lands on the
    # right OpenAI-compatible endpoint.
    local payload
    payload=$(jq -n --arg name "$model_name" --arg base "$model_url" '{
        model_name: $name,
        litellm_params: { model: ("openai/\($name)"), api_base: $base, api_key: "unused" }
    }')
    info "Registering $model_name with LiteLLM (api_base=$model_url)"

    # Retry, then verify via /model/info. Registration is the final step of a
    # deploy and the POST is not guaranteed: the proxy may still be rolling or
    # reconnecting to Postgres when the model goes Ready. Swallowing the failure
    # (the old `|| true`) leaves a running model absent from /v1/models with
    # nothing to retry it and no diagnostic.
    local attempt
    for attempt in 1 2 3; do
        if litellm_curl POST "$litellm_base/model/new" "$payload" >/dev/null 2>&1 &&
           [[ -n "$(litellm_model_ids "$model_name" "$litellm_base")" ]]; then
            ok "Registered with LiteLLM"
            return 0
        fi
        (( attempt < 3 )) && { warn "LiteLLM registration attempt $attempt failed — retrying"; sleep $(( attempt * 5 )); }
    done

    warn "Could not register $model_name with LiteLLM after 3 attempts."
    warn "  The model is serving but will NOT appear in /v1/models."
    warn "  Retry with: ./model-manager deploy $model_name"
    return 0
}

litellm_deregister() {
    local model_name="$1"
    litellm_enabled || return 0
    local url; url=$(litellm_url)
    [[ -z "$url" ]] && return 0

    # LiteLLM's /model/delete keys on the UUID it assigned at registration
    # (model_info.id), NOT the human model_name. Passing the name is a silent
    # no-op — the model lingers. So resolve every UUID registered under this
    # name via /model/info, then delete each. (Re-registers can create dupes,
    # so handle more than one.)
    local ids id
    ids=$(litellm_model_ids "$model_name" "$url")

    if [[ -z "$ids" ]]; then
        # Fall back to deleting by name in case an older entry was stored that way.
        litellm_curl POST "$url/model/delete" "{\"id\":\"$model_name\"}" >/dev/null 2>&1 || true
        return 0
    fi

    # Iterate ids without a `while read` loop: litellm_curl runs `kubectl run -i`,
    # which consumes the loop's stdin and eats subsequent iterations. A for-loop
    # over word-split ids avoids that entirely (ids are UUIDs — no whitespace).
    for id in $ids; do
        [[ -z "$id" ]] && continue
        litellm_curl POST "$url/model/delete" "{\"id\":\"$id\"}" >/dev/null 2>&1 || true
    done
}

# ── AI Gateway ──

AI_GATEWAY_NAME="${MM_AI_GATEWAY_NAME:-ai-gateway}"
AI_GATEWAY_NS="${MM_AI_GATEWAY_NS:-llm-inference}"
ROUTE_TIMEOUT="${MM_AIGATEWAY_ROUTE_TIMEOUT:-600s}"

# Which AI gateway data plane is deployed. Both share the same Gateway name and
# unified URL; they differ only in the routing CRDs model-manager must emit:
#   envoy        → AIGatewayRoute (aigateway.envoyproxy.io) + auto-injected
#                  x-ai-eg-model header for dispatch.
#   agentgateway → standard Gateway API HTTPRoute matched on the
#                  X-Gateway-Base-Model-Name header (populated cluster-side by
#                  the extract-model-header AgentgatewayPolicy).
# ai_gateway_provider lives in the inference solution config
# (config.inference.yaml) — its canonical home. It may also be overridden in
# global_config.yaml or via the env var.
# Precedence: MM_AI_GATEWAY_PROVIDER env
#             → config.inference.yaml → global_config.yaml → envoy.
ai_gateway_provider() {
    local val="${MM_AI_GATEWAY_PROVIDER:-}"
    [[ "$val" == "agentgateway" ]] && { echo "agentgateway"; return; }
    [[ "$val" == "envoy" ]] && { echo "envoy"; return; }

    local cfg provider
    for cfg in "$(discover_inference_config 2>/dev/null)" "$(discover_global_config 2>/dev/null)"; do
        [[ -n "$cfg" && -f "$cfg" ]] || continue
        provider=$(yq e '.ai_gateway_provider // ""' "$cfg" 2>/dev/null)
        [[ "$provider" == "agentgateway" ]] && { echo "agentgateway"; return; }
        [[ "$provider" == "envoy" ]] && { echo "envoy"; return; }
    done
    echo "envoy"
}

# The AI Gateway sits in front of every deploy — Keycloak enforces JWT on it,
# LiteLLM enforces its virtual keys. Each mode serves every model on one unified
# URL, but on its own host (inference.<domain> vs litellm.<domain>).
# Default: enabled. MM_AIGATEWAY_ENABLED=false skips gateway route creation entirely.
aigateway_enabled() {
    local val="${MM_AIGATEWAY_ENABLED:-}"
    [[ "$val" =~ ^(false|0|no)$ ]] && return 1
    return 0
}

effective_routing() {
    local model_routing="$1" model_kind="$2"
    if [[ -n "$model_routing" ]]; then echo "$model_routing"
    elif [[ "$model_kind" == "LLMInferenceService" ]]; then echo "epp"
    else echo "direct"; fi
}

# Pick the direct-route template for a (kind, category, provider).
#
# Three routing modes:
#   direct    — plain HTTPRoute; client must supply x-ai-eg-model header.
#   epp       — AIGatewayRoute → InferencePool (vLLM KV-cache-aware LB).
#   aigateway — AIGatewayRoute → AIServiceBackend → Backend; ext-proc auto-
#               extracts model from request body (no explicit header needed).
#
# This function only handles the "direct" path. "epp" and "aigateway" are
# handled inline in aigateway_register().
#
# vLLM (LLMInferenceService) embed/rerank ride OpenAI/extension paths the gateway
# ext-proc can't parse for model extraction — those templates match by path and
# inject x-ai-eg-model server-side. vLLM LLM keeps the header-matched catch-all.
#
# agentgateway path: routing is driven entirely by the X-Gateway-Base-Model-Name
# header (set by the extract-model-header PreRouting policy) — everything is a
# plain header-matched passthrough.
direct_template() {
    local kind="$1" category="$2" provider="${3:-envoy}"
    if [[ "$provider" == "agentgateway" ]]; then
        if [[ "$kind" == "InferenceService" ]]; then
            echo "$MM_TEMPLATES/agw-route-direct-isvc.yaml"
        else
            echo "$MM_TEMPLATES/agw-route-direct-llm.yaml"
        fi
        return
    fi
    if [[ "$kind" == "InferenceService" ]]; then
        echo "$MM_TEMPLATES/httproute-direct-isvc.yaml"
        return
    fi
    case "$category" in
        embed)  echo "$MM_TEMPLATES/httproute-direct-embed.yaml" ;;
        rerank) echo "$MM_TEMPLATES/httproute-direct-rerank.yaml" ;;
        *)      echo "$MM_TEMPLATES/httproute-direct-llm.yaml" ;;
    esac
}

aigateway_register() {
    local model_name="$1" namespace="$2" kind="$3" routing="$4" category="${5:-}"
    aigateway_enabled || return 0
    routing=$(effective_routing "$routing" "$kind")
    local provider; provider=$(ai_gateway_provider)

    if [[ "$routing" == "epp" ]]; then
        local pool_name="${model_name}-inference-pool"
        local tmpl
        if [[ "$provider" == "agentgateway" ]]; then
            tmpl="$MM_TEMPLATES/agw-route-pool.yaml"
            info "Creating HTTPRoute aigw-$model_name (agentgateway → InferencePool $pool_name)"
        else
            tmpl="$MM_TEMPLATES/aigateway-route.yaml"
            info "Creating AIGatewayRoute aigw-$model_name (EPP → $pool_name)"
        fi
        kube_ssa "$(render_template "$tmpl" \
            "MODEL_NAME=$model_name" "NAMESPACE=$namespace" \
            "POOL_NAME=$pool_name" \
            "ROUTE_TIMEOUT=$ROUTE_TIMEOUT" \
            "AI_GATEWAY_NAME=$AI_GATEWAY_NAME" "AI_GATEWAY_NAMESPACE=$AI_GATEWAY_NS")"
    elif [[ "$routing" == "aigateway" ]]; then
        local svc_name svc_port
        if [[ "$kind" == "InferenceService" ]]; then
            svc_name="${model_name}-predictor"
            svc_port=80
        else
            svc_name="${model_name}-kserve-workload-svc"
            svc_port=8000
        fi
        if [[ "$provider" == "agentgateway" ]]; then
            # agentgateway already extracts model from body via PreRouting policy;
            # use the standard direct HTTPRoute (header-matched passthrough).
            local tmpl=$(direct_template "$kind" "$category" "$provider")
            info "Creating HTTPRoute direct-$model_name (agentgateway direct → $svc_name:$svc_port)"
            kube_ssa "$(render_template "$tmpl" \
                "MODEL_NAME=$model_name" "NAMESPACE=$namespace" \
                "SERVICE_NAME=$svc_name" "SERVICE_PORT=$svc_port" \
                "ROUTE_TIMEOUT=$ROUTE_TIMEOUT" \
                "AI_GATEWAY_NAME=$AI_GATEWAY_NAME" "AI_GATEWAY_NAMESPACE=$AI_GATEWAY_NS")"
        else
            info "Creating AIGatewayRoute aigw-$model_name (aigateway → $svc_name:$svc_port)"
            kube_ssa "$(render_template "$MM_TEMPLATES/aigateway-route-direct.yaml" \
                "MODEL_NAME=$model_name" "NAMESPACE=$namespace" \
                "SERVICE_NAME=$svc_name" "SERVICE_PORT=$svc_port" \
                "ROUTE_TIMEOUT=$ROUTE_TIMEOUT" \
                "AI_GATEWAY_NAME=$AI_GATEWAY_NAME" "AI_GATEWAY_NAMESPACE=$AI_GATEWAY_NS")"
        fi
    else
        local svc_name svc_port tmpl
        if [[ "$kind" == "InferenceService" ]]; then
            svc_name="${model_name}-predictor"
            svc_port=80
        else
            svc_name="${model_name}-kserve-workload-svc"
            svc_port=8000
        fi
        tmpl=$(direct_template "$kind" "$category" "$provider")
        info "Creating HTTPRoute direct-$model_name (direct → $svc_name:$svc_port, ${category:-llm})"
        kube_ssa "$(render_template "$tmpl" \
            "MODEL_NAME=$model_name" "NAMESPACE=$namespace" \
            "SERVICE_NAME=$svc_name" "SERVICE_PORT=$svc_port" \
            "ROUTE_TIMEOUT=$ROUTE_TIMEOUT" \
            "AI_GATEWAY_NAME=$AI_GATEWAY_NAME" "AI_GATEWAY_NAMESPACE=$AI_GATEWAY_NS")"
    fi
}

aigateway_deregister() {
    local model_name="$1" namespace="$2" kind="$3" routing="$4"
    aigateway_enabled || return 0
    routing=$(effective_routing "$routing" "$kind")
    local provider; provider=$(ai_gateway_provider)

    if [[ "$routing" == "epp" ]]; then
        if [[ "$provider" == "agentgateway" ]]; then
            kubectl delete httproute "aigw-$model_name" -n "$namespace" --ignore-not-found 2>/dev/null || true
        else
            kubectl delete aigatewayroute "aigw-$model_name" -n "$namespace" --ignore-not-found 2>/dev/null || true
        fi
    elif [[ "$routing" == "aigateway" ]]; then
        if [[ "$provider" == "agentgateway" ]]; then
            kubectl delete httproute "direct-$model_name" -n "$namespace" --ignore-not-found 2>/dev/null || true
        else
            kubectl delete aigatewayroute "aigw-$model_name" -n "$namespace" --ignore-not-found 2>/dev/null || true
            kubectl delete aiservicebackend "$model_name" -n "$namespace" --ignore-not-found 2>/dev/null || true
            kubectl delete backend "$model_name" -n "$namespace" --ignore-not-found 2>/dev/null || true
        fi
    else
        kubectl delete httproute "direct-$model_name" -n "$namespace" --ignore-not-found 2>/dev/null || true
    fi
}
