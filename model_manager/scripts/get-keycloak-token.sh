#!/usr/bin/env bash
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# get-keycloak-token.sh — Fetch a Keycloak JWT token for inference API access
# Usage: source ./scripts/get-keycloak-token.sh [--lifespan SECONDS]
#   Exports: TOKEN, GATEWAY_IP, GATEWAY_DOMAIN, GATEWAY_URL (API base, path-aware)
#   All values auto-discovered from cluster — no manual config needed.
#
# Options:
#   --lifespan SECONDS   Set token validity (updates Keycloak client setting)
#                        Examples: --lifespan 3600 (1 hour), --lifespan 86400 (24 hours)

# Guard: when sourced, set -e would kill the parent shell on any failure.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  # Sourced — save and restore shell options
  _gt_oldopts=$(set +o); set +e; set +u; set +o pipefail
else
  set -euo pipefail
fi
_gt_cleanup() { if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then eval "$_gt_oldopts"; unset _gt_oldopts; fi; }
trap _gt_cleanup RETURN 2>/dev/null || true

_gt_err() { echo "ERROR: $*" >&2; return 1 2>/dev/null || exit 1; }

# ── Parse arguments ──────────────────────────────────────────────────────────
_gt_lifespan=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lifespan|-l)
      _gt_lifespan="$2"; shift 2 ;;
    --lifespan=*|-l=*)
      _gt_lifespan="${1#*=}"; shift ;;
    -h|--help)
      echo "Usage: source ./scripts/get-keycloak-token.sh [--lifespan SECONDS]"
      echo "  --lifespan, -l   Set JWT token validity in seconds (updates Keycloak client)"
      echo "                   Examples: 3600 (1h), 86400 (24h), 900 (15min, default)"
      return 0 2>/dev/null || exit 0 ;;
    *)
      _gt_err "Unknown option: $1. Use --help for usage." ;;
  esac
done

# ── Proxy normalisation (reuse installer's env helper if present) ───────────
# Walks up from wherever this script lives until it finds es_auto_installer.sh,
# so the same file works from model_manager/scripts/ or a symlink elsewhere.
_gt_script_dir="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" && pwd)"
_gt_search="$_gt_script_dir"
for _ in 1 2 3 4 5; do
  if [[ -x "$_gt_search/es_auto_installer.sh" ]]; then
    eval "$("$_gt_search/es_auto_installer.sh" env 2>/dev/null)" 2>/dev/null || true
    break
  fi
  _gt_search=$(dirname "$_gt_search")
  [[ "$_gt_search" == "/" ]] && break
done
unset _gt_script_dir _gt_search

# ── Preflight checks ──────────────────────────────────────────────────────────
if ! command -v kubectl &>/dev/null; then
  _gt_err "kubectl not found in PATH. Install it or ensure it is on PATH."
fi

if ! command -v python3 &>/dev/null; then
  _gt_err "python3 not found. Install python3 to parse the token response."
fi

if ! kubectl cluster-info --request-timeout=5s &>/dev/null; then
  _gt_err "kubectl cannot reach the cluster. Check KUBECONFIG or run:\n  export KUBECONFIG=~/.kube/config"
fi

# ─────────────────────────────────────────────────────────────────────────────

# Discover gateway IP from cluster (filter for LoadBalancer type to skip ClusterIP services)
GATEWAY_IP=$(kubectl get svc -n envoy-gateway-system \
  -l gateway.envoyproxy.io/owning-gateway-name \
  --field-selector spec.type=LoadBalancer \
  -o jsonpath='{.items[0].status.loadBalancer.ingress[0].ip}' 2>/dev/null)

if [[ -z "$GATEWAY_IP" ]]; then
  _gt_err "Could not find Envoy Gateway LoadBalancer IP.\n  Is the platform installed? Run: ./es_auto_installer.sh install platform"
fi

# Discover domains from Keycloak CR
KEYCLOAK_DOMAIN=$(kubectl get keycloaks -n keycloak -o jsonpath='{.items[0].spec.hostname.hostname}' 2>/dev/null)
if [[ -z "$KEYCLOAK_DOMAIN" ]]; then
  _gt_err "Could not read Keycloak hostname from cluster.\n  Is Keycloak installed? Run: ./es_auto_installer.sh install keycloak"
fi
# Strip protocol prefix if present (handles both http:// and https://)
KEYCLOAK_DOMAIN="${KEYCLOAK_DOMAIN#http://}"
KEYCLOAK_DOMAIN="${KEYCLOAK_DOMAIN#https://}"
# Path routing: the frontend URL is <host>/<prefix>; split it.
KEYCLOAK_PATH=""
if [[ "$KEYCLOAK_DOMAIN" == */* ]]; then
  KEYCLOAK_PATH="/${KEYCLOAK_DOMAIN#*/}"
  KEYCLOAK_PATH="${KEYCLOAK_PATH%/}"
  KEYCLOAK_DOMAIN="${KEYCLOAK_DOMAIN%%/*}"
fi

# Read client credentials from cluster
CLIENT_ID=$(kubectl get secret keycloak-client-secret -n keycloak \
  -o jsonpath='{.data.client-id}' 2>/dev/null | base64 -d 2>/dev/null)
CLIENT_SECRET=$(kubectl get secret keycloak-client-secret -n keycloak \
  -o jsonpath='{.data.client-secret}' 2>/dev/null | base64 -d 2>/dev/null)

if [[ -z "$CLIENT_ID" || -z "$CLIENT_SECRET" ]]; then
  _gt_err "Secret 'keycloak-client-secret' not found in namespace 'keycloak'.\n  Run the inference stack install to create it:\n  ./es_auto_installer.sh install inference"
fi

# ── Keycloak reachability: prefer direct (external), fall back to port-forward ─
# Sets _gt_kc_base (the URL prefix to prepend to /realms/... or /admin/...)
# and _gt_kc_curl_extra (extra curl args like --resolve for the direct path).
# When port-forwarding, we speak HTTPS to keycloak-service:https (8443).
_gt_open_keycloak() {
  local resolve_ip="$GATEWAY_IP"
  # Attempt 1: direct connection through external LB.
  # Require HTTP 200 AND a JSON body — some clusters proxy 5xx errors as HTML
  # through the same route, which would otherwise pass a naive probe.
  local probe_body probe_code
  probe_body=$(curl -sk --noproxy '*' --connect-timeout 3 --max-time 5 \
      --resolve "${KEYCLOAK_DOMAIN}:443:${resolve_ip}" \
      -o /dev/stdout -w "\n__HTTP__%{http_code}" \
      "https://${KEYCLOAK_DOMAIN}${KEYCLOAK_PATH}/realms/inference/.well-known/openid-configuration" 2>/dev/null)
  probe_code="${probe_body##*__HTTP__}"
  probe_body="${probe_body%__HTTP__*}"
  if [[ "$probe_code" == "200" && "$probe_body" == *'"issuer"'* ]]; then
    _gt_kc_base="https://${KEYCLOAK_DOMAIN}${KEYCLOAK_PATH}"
    _gt_kc_curl_extra=(--resolve "${KEYCLOAK_DOMAIN}:443:${resolve_ip}")
    _gt_kc_pf_pid=""
    return 0
  fi

  # Attempt 2: port-forward keycloak-service on port `https` (8443).
  local kc_ns="${MM_KEYCLOAK_NS:-keycloak}"
  local kc_svc="${MM_KEYCLOAK_SVC:-keycloak-service}"
  local local_port
  local_port=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
  kubectl port-forward -n "$kc_ns" "svc/${kc_svc}" "${local_port}:https" &>/dev/null &
  _gt_kc_pf_pid=$!
  sleep 2
  if ! kill -0 "$_gt_kc_pf_pid" 2>/dev/null; then
    _gt_kc_pf_pid=""
    _gt_err "Cannot reach Keycloak — port-forward to ${kc_ns}/${kc_svc} failed."
  fi
  _gt_kc_base="https://127.0.0.1:${local_port}${KEYCLOAK_PATH}"
  _gt_kc_curl_extra=()
}

_gt_close_keycloak() {
  if [[ -n "${_gt_kc_pf_pid:-}" ]]; then
    kill "$_gt_kc_pf_pid" 2>/dev/null
    wait "$_gt_kc_pf_pid" 2>/dev/null
    unset _gt_kc_pf_pid
  fi
  unset _gt_kc_base _gt_kc_curl_extra
}

_gt_open_keycloak

# ── Update token lifespan if requested ────────────────────────────────────────
if [[ -n "$_gt_lifespan" ]]; then
  echo "  Updating token lifespan to ${_gt_lifespan}s..."

  _gt_admin_user=$(kubectl get secret keycloak-admin-secret -n keycloak \
    -o jsonpath='{.data.username}' 2>/dev/null | base64 -d 2>/dev/null)
  _gt_admin_pass=$(kubectl get secret keycloak-admin-secret -n keycloak \
    -o jsonpath='{.data.password}' 2>/dev/null | base64 -d 2>/dev/null)
  if [[ -z "$_gt_admin_user" ]]; then
    _gt_admin_user=$(kubectl get secret keycloak-initial-admin -n keycloak \
      -o jsonpath='{.data.username}' 2>/dev/null | base64 -d 2>/dev/null)
    _gt_admin_pass=$(kubectl get secret keycloak-initial-admin -n keycloak \
      -o jsonpath='{.data.password}' 2>/dev/null | base64 -d 2>/dev/null)
  fi

  if [[ -z "$_gt_admin_user" || -z "$_gt_admin_pass" ]]; then
    _gt_close_keycloak
    _gt_err "Could not read Keycloak admin credentials (keycloak-admin-secret or keycloak-initial-admin)."
  fi

  # Get master admin token via the opened Keycloak channel (direct or port-forward).
  _gt_admin_token=$(curl -sk --noproxy '*' --connect-timeout 5 --max-time 15 \
    "${_gt_kc_curl_extra[@]}" \
    "${_gt_kc_base}/realms/master/protocol/openid-connect/token" \
    -d "grant_type=password" \
    -d "client_id=admin-cli" \
    -d "username=${_gt_admin_user}" \
    -d "password=${_gt_admin_pass}" 2>/dev/null | \
    python3 -c "import json,sys; print(json.load(sys.stdin).get('access_token',''))" 2>/dev/null)

  if [[ -z "$_gt_admin_token" ]]; then
    _gt_close_keycloak
    _gt_err "Could not get Keycloak admin token. Check admin credentials."
  fi

  _gt_client_uuid=$(curl -sk --noproxy '*' --connect-timeout 5 --max-time 15 \
    "${_gt_kc_curl_extra[@]}" \
    -H "Authorization: Bearer ${_gt_admin_token}" \
    "${_gt_kc_base}/admin/realms/inference/clients?clientId=${CLIENT_ID}" 2>/dev/null | \
    python3 -c "import json,sys; clients=json.load(sys.stdin); print(clients[0]['id'] if clients else '')" 2>/dev/null)

  if [[ -z "$_gt_client_uuid" ]]; then
    _gt_close_keycloak
    _gt_err "Could not find client '${CLIENT_ID}' in Keycloak realm 'inference'."
  fi

  _gt_update_resp=$(curl -sk --noproxy '*' --connect-timeout 5 --max-time 15 \
    "${_gt_kc_curl_extra[@]}" \
    -X PUT \
    -H "Authorization: Bearer ${_gt_admin_token}" \
    -H "Content-Type: application/json" \
    -d "{\"attributes\":{\"access.token.lifespan\":\"${_gt_lifespan}\"}}" \
    -o /dev/null -w "%{http_code}" \
    "${_gt_kc_base}/admin/realms/inference/clients/${_gt_client_uuid}" 2>/dev/null)

  if [[ "$_gt_update_resp" == "204" || "$_gt_update_resp" == "200" ]]; then
    echo "  Token lifespan updated to ${_gt_lifespan}s ($((_gt_lifespan / 60)) min)"
  else
    echo "  WARNING: Failed to update lifespan (HTTP ${_gt_update_resp}). Using current server setting." >&2
  fi

  unset _gt_admin_user _gt_admin_pass _gt_admin_token _gt_client_uuid _gt_update_resp
fi
unset _gt_lifespan

# ── Request JWT token via the opened Keycloak channel ────────────────────────
_gt_response=$(curl -sk --noproxy '*' --connect-timeout 5 --max-time 15 \
  "${_gt_kc_curl_extra[@]}" \
  "${_gt_kc_base}/realms/inference/protocol/openid-connect/token" \
  -d "grant_type=client_credentials" \
  -d "client_id=${CLIENT_ID}" \
  -d "client_secret=${CLIENT_SECRET}" 2>/dev/null)

TOKEN=$(echo "$_gt_response" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('access_token',''))" 2>/dev/null)
_gt_close_keycloak

if [[ -z "$TOKEN" ]]; then
  echo "ERROR: Failed to get token from Keycloak." >&2
  echo "  Keycloak response: $_gt_response" >&2
  return 1 2>/dev/null || exit 1
fi

# Derive the inference endpoint from Keycloak's: <base>/inference in path mode,
# else inference.<base>. GATEWAY_URL is the base for /v1/... requests.
if [[ -n "$KEYCLOAK_PATH" ]]; then
  GATEWAY_DOMAIN="$KEYCLOAK_DOMAIN"
  GATEWAY_URL="https://${GATEWAY_DOMAIN}/inference"
else
  GATEWAY_DOMAIN="inference.${KEYCLOAK_DOMAIN#keycloak.}"
  GATEWAY_URL="https://${GATEWAY_DOMAIN}"
fi

# Show token expiry
_gt_exp=$(echo "$TOKEN" | python3 -c "
import sys, json, base64
t = sys.stdin.read().strip().split('.')[1]
t += '=' * (4 - len(t) % 4)
exp = json.loads(base64.urlsafe_b64decode(t))['exp']
iat = json.loads(base64.urlsafe_b64decode(t)).get('iat', exp)
print(f'{exp - iat}')
" 2>/dev/null)

export TOKEN GATEWAY_IP GATEWAY_DOMAIN GATEWAY_URL KEYCLOAK_DOMAIN
echo "Token acquired. Exported: TOKEN, GATEWAY_IP, GATEWAY_DOMAIN, GATEWAY_URL"
echo "  Gateway:  ${GATEWAY_URL} → ${GATEWAY_IP}"
echo "  Keycloak: ${KEYCLOAK_DOMAIN}${KEYCLOAK_PATH}"
if [[ -n "${_gt_exp:-}" ]]; then
  echo "  Validity: ${_gt_exp}s ($((_gt_exp / 60)) min)"
fi
unset _gt_exp
