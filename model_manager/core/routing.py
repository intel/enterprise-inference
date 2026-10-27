"""Gateway routes, InferencePool acceptance, LiteLLM registration and endpoint hints."""
from __future__ import annotations

import base64
import binascii
import json
import os
import secrets
import time

from . import k8s
from .config import MM_DIR, Model, config_value, env, gateway_ref, gateway_service, global_config, inference_config
from .renderer import render, template
from .ui import BOLD, CYAN, DIM, RESET, info, ok, warn


TRUE, FALSE = ("true", "1", "yes"), ("false", "0", "no")


def auth_provider() -> str:
    """litellm or keycloak: MM_LITELLM_ENABLED, else auth_provider from global_config.yaml."""
    v = os.environ.get("MM_LITELLM_ENABLED", "")
    if v in TRUE or v in FALSE:
        return "litellm" if v in TRUE else "keycloak"
    return "litellm" if config_value(global_config(), "auth_provider") == "litellm" else "keycloak"


def gateway_provider() -> str:
    """envoy (AIGatewayRoute) or agentgateway (HTTPRoute): env, then config.inference.yaml, then global config."""
    for value in (os.environ.get("MM_AI_GATEWAY_PROVIDER", ""),
                  *(config_value(c, "ai_gateway_provider") for c in (inference_config(), global_config()))):
        if value in ("envoy", "agentgateway"):
            return value
    return "envoy"


def gateway_enabled() -> bool:
    return os.environ.get("MM_AIGATEWAY_ENABLED", "") not in FALSE


def _service(m: Model) -> tuple[str, str]:
    if m.kind == "InferenceService":
        return f"{m.name}-predictor", "80"
    return f"{m.name}-kserve-workload-svc", "8000"


def route(m: Model) -> tuple[str, list[tuple[str, str]], str]:
    """(template, [(kind, name)] it creates, description) for m's routing mode and gateway provider.

    epp       AIGatewayRoute/HTTPRoute -> InferencePool (KV-cache aware endpoint picking)
    aigateway AIGatewayRoute -> AIServiceBackend (model read from the body); plain HTTPRoute on agentgateway
    direct    HTTPRoute to the workload service, matched on the model header
    """
    routing = m.routing or ("epp" if m.kind == "LLMInferenceService" else "direct")
    agw, isvc, (svc, port) = gateway_provider() == "agentgateway", m.kind == "InferenceService", _service(m)
    if routing == "epp":
        pool = f"{m.name}-inference-pool"
        if agw:
            return "agw-route-pool.yaml", [("httproute", f"aigw-{m.name}")], \
                f"HTTPRoute aigw-{m.name} (agentgateway → InferencePool {pool})"
        return "aigateway-route.yaml", [("aigatewayroute", f"aigw-{m.name}")], \
            f"AIGatewayRoute aigw-{m.name} (EPP → {pool})"
    if routing == "aigateway" and not agw:
        return "aigateway-route-direct.yaml", [("aigatewayroute", f"aigw-{m.name}"), ("aiservicebackend", m.name),
                                               ("backend", m.name)], f"AIGatewayRoute aigw-{m.name} (aigateway → {svc}:{port})"
    if agw:
        tmpl = f"agw-route-direct-{'isvc' if isvc else 'llm'}.yaml"
    else:
        tmpl = f"httproute-direct-{'isvc' if isvc else m.category if m.category in ('embed', 'rerank') else 'llm'}.yaml"
    via = "agentgateway direct" if routing == "aigateway" else "direct"
    detail = f"{svc}:{port}" if routing == "aigateway" else f"{svc}:{port}, {m.category or 'llm'}"
    return tmpl, [("httproute", f"direct-{m.name}")], f"HTTPRoute direct-{m.name} ({via} → {detail})"


def register(m: Model) -> None:
    if not gateway_enabled():
        return
    tmpl, _, description = route(m)
    svc, port = _service(m)
    info(f"Creating {description}")
    k8s.apply(render(template(tmpl), MODEL_NAME=m.name, NAMESPACE=m.namespace, POOL_NAME=f"{m.name}-inference-pool",
                     SERVICE_NAME=svc, SERVICE_PORT=port, ROUTE_TIMEOUT=env("MM_AIGATEWAY_ROUTE_TIMEOUT", "600s"),
                     AI_GATEWAY_NAME=env("MM_AI_GATEWAY_NAME", "ai-gateway"),
                     AI_GATEWAY_NAMESPACE=env("MM_AI_GATEWAY_NS", "llm-inference")), ssa=True)


def deregister(m: Model) -> None:
    if gateway_enabled():
        for kind, name in route(m)[1]:
            k8s.delete(kind, name, m.namespace)


def patch_pool_status(m: Model) -> None:
    """Mark the InferencePool Accepted so the gateway route goes live (done at ready time, after KServe creates it)."""
    pool = f"{m.name}-inference-pool"
    for _ in range(24):
        if k8s.succeeds("get", "inferencepool", pool, "-n", m.namespace):
            break
        time.sleep(5)
    else:
        return
    ref = "jsonpath={.spec.router.gateway.refs[0].%s}"
    gw = k8s.out("get", "llminferenceservice", m.name, "-n", m.namespace, "-o", ref % "name")
    gw_ns = k8s.out("get", "llminferenceservice", m.name, "-n", m.namespace, "-o", ref % "namespace")
    ref_gw, ref_ns = gateway_ref()
    controller = ("agentgateway.dev/agentgateway" if gateway_provider() == "agentgateway"
                  else "gateway.envoyproxy.io/gatewayclass-controller")
    status = {"parents": [{
        "parentRef": {"group": "gateway.networking.k8s.io", "kind": "Gateway",
                      "name": gw or ref_gw, "namespace": gw_ns or ref_ns},
        "controllerName": controller,
        "conditions": [{"type": "Accepted", "status": "True", "reason": "Accepted", "message": "Patched by model-manager",
                        "lastTransitionTime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}]}]}
    k8s.run("patch", "inferencepool", pool, "-n", m.namespace, "--type=merge", "--subresource=status",
            "-p", json.dumps({"status": status}, separators=(",", ":")))


# ── LiteLLM ───────────────────────────────────────────────────────────────────

def litellm_enabled() -> bool:
    return auth_provider() == "litellm"


def _internal_url(m: Model) -> str:
    """api_base LiteLLM calls: the AI Gateway for vLLM (EPP routing), the predictor for OVMS.
    MM_LITELLM_KSERVE_URL overrides the gateway only: OVMS has no route there with the AI Gateway off."""
    if m.kind == "InferenceService":
        port = k8s.out("get", "svc", f"{m.name}-predictor", "-n", m.namespace, "-o", "jsonpath={.spec.ports[0].port}")
        return f"http://{m.name}-predictor.{m.namespace}.svc.cluster.local:{port or 80}/v1"
    if os.environ.get("MM_LITELLM_KSERVE_URL"):
        return f"{os.environ['MM_LITELLM_KSERVE_URL'].rstrip('/')}/{m.namespace}/{m.name}/v1"
    gw, gw_ns = gateway_service()
    if k8s.out("get", "svc", gw, "-n", gw_ns, "-o", "jsonpath={.metadata.name}"):
        return f"http://{gw}.{gw_ns}.svc.cluster.local/{m.namespace}/{m.name}/v1"
    warn(f"AI Gateway service '{gw}' not found in {gw_ns} — LiteLLM will bypass gateway for {m.name}")
    return f"http://{m.name}-kserve-workload-svc.{m.namespace}.svc.cluster.local:8000/v1"


def litellm_ns() -> str:
    return env("MM_LITELLM_NAMESPACE", "litellm")


def litellm_key_ref() -> str:
    """Name of the Secret holding the master key (the value itself is never printed)."""
    return env("MM_LITELLM_SECRET", "litellm-master-key")


def _litellm_base() -> str:
    if os.environ.get("MM_LITELLM_URL"):
        return os.environ["MM_LITELLM_URL"].rstrip("/")
    svc, ns = env("MM_LITELLM_SERVICE", "litellm"), litellm_ns()
    port = k8s.out("get", "svc", svc, "-n", ns, "-o", "jsonpath={.spec.ports[0].port}")
    return f"http://{svc}.{ns}.svc.cluster.local:{port}" if port else ""


def _litellm_key() -> str:
    if os.environ.get("MM_LITELLM_MASTER_KEY"):
        return os.environ["MM_LITELLM_MASTER_KEY"]
    encoded = k8s.out("get", "secret", litellm_key_ref(), "-n", litellm_ns(), "-o", "jsonpath={.data.master_key}")
    try:
        return base64.b64decode(encoded).decode().strip()
    except (binascii.Error, UnicodeDecodeError):
        return ""


def _litellm(method: str, url: str, payload: dict | None = None) -> str | None:
    """Call LiteLLM from a throwaway in-cluster curl pod; the key goes in on stdin, never in the pod spec."""
    key = _litellm_key()
    curl = ["curl", "-sf", "-X", method, "-H", "Content-Type: application/json", *(["-H", "@-"] if key else []),
            *(["-d", json.dumps(payload)] if payload else []), url]
    pod = f"mm-curl-{secrets.token_hex(4)}"
    # Restricted Pod Security; curlimages/curl runs as the non-root curl_user (uid 100).
    restricted = {"apiVersion": "v1", "spec": {
        "securityContext": {"runAsNonRoot": True, "runAsUser": 100, "seccompProfile": {"type": "RuntimeDefault"}},
        "containers": [{"name": pod, "securityContext": {"allowPrivilegeEscalation": False,
                                                         "capabilities": {"drop": ["ALL"]}}}]}}
    r = k8s.run("run", pod, "--rm", "-i", "--quiet", "--restart=Never", "-n", litellm_ns(),
                f"--image={env('MM_CURL_IMAGE', 'curlimages/curl:8.12.1')}", "--override-type=strategic",
                f"--overrides={json.dumps(restricted)}", "--", *curl, stdin=f"Authorization: Bearer {key}\n" if key else "")
    return r.stdout if r.returncode == 0 else None  # --quiet: stdout is curl's response only


def _litellm_ids(name: str, base: str) -> list[str]:
    """LiteLLM keys models by the UUID it minted, so map name -> ids via /model/info."""
    try:
        data = json.loads(_litellm("GET", f"{base}/model/info") or "{}")
    except ValueError:
        return []
    return [d["model_info"]["id"] for d in data.get("data") or []
            if d.get("model_name") == name and (d.get("model_info") or {}).get("id")]


def litellm_register(m: Model) -> None:
    if not litellm_enabled():
        return
    base = _litellm_base()
    if not base:
        return warn("LiteLLM not reachable — skipping registration")
    litellm_deregister(m.name)  # /model/new always mints a new id, so clear the name first
    url = _internal_url(m)
    mode = {"embed": "embedding", "rerank": "rerank"}.get(m.category)
    # LiteLLM's openai provider has no rerank and sends encoding_format: null on embeddings
    provider = "hosted_vllm" if mode and m.runtime == "vllm" else "openai"
    payload = {"model_name": m.name, "litellm_params": {"model": f"{provider}/{m.name}", "api_base": url, "api_key": "unused"}}
    if mode:
        payload["model_info"] = {"mode": mode}
    info(f"Registering {m.name} with LiteLLM (api_base={url})")
    for attempt in (1, 2, 3):  # the proxy may still be rolling out when the model turns Ready
        if _litellm("POST", f"{base}/model/new", payload) is not None and _litellm_ids(m.name, base):
            return ok("Registered with LiteLLM")
        if attempt < 3:
            warn(f"LiteLLM registration attempt {attempt} failed — retrying")
            time.sleep(attempt * 5)
    warn(f"Could not register {m.name} with LiteLLM after 3 attempts.")
    warn("  The model is serving but will NOT appear in /v1/models.")
    warn(f"  Retry with: ./model-manager deploy {m.name}")


def litellm_deregister(name: str) -> None:
    base = _litellm_base() if litellm_enabled() else ""
    if not base:
        return
    for model_id in _litellm_ids(name, base) or [name]:  # older entries may be keyed by name
        _litellm("POST", f"{base}/model/delete", {"id": model_id})


# ── Endpoint hints ────────────────────────────────────────────────────────────

def inference_hostname() -> str:
    """API host: inference.<domain> under keycloak, litellm.<domain> (LiteLLM's own route) under litellm."""
    if os.environ.get("MM_INFERENCE_HOSTNAME"):
        return os.environ["MM_INFERENCE_HOSTNAME"]
    gc, prefix = global_config(), ""
    if litellm_enabled():
        for cfg in (inference_config(), gc):
            if config_value(cfg, "litellm_hostname"):
                return config_value(cfg, "litellm_hostname")
        prefix = "litellm"
    if gc:
        if not prefix and config_value(gc, "inference_hostname"):
            return config_value(gc, "inference_hostname")
        prefix = prefix or config_value(gc, "inference_prefix") or "inference"
        domain = config_value(gc, "gateway_domain") or config_value(gc, "base_domain_name")
        if domain:
            return f"{prefix}.{domain}"
    warn("Could not resolve inference hostname — set MM_INFERENCE_HOSTNAME or MM_GLOBAL_CONFIG")
    return ""


def print_endpoints(m: Model) -> None:
    host, litellm = inference_hostname(), litellm_enabled()
    gw_ip = k8s.out("get", "svc", "-n", env("MM_EG_GATEWAY_NS", "envoy-gateway-system"), "-l",
                    f"gateway.envoyproxy.io/owning-gateway-name={env('MM_EG_GATEWAY_NAME', 'eg-gateway')}",
                    "-o", "jsonpath={.items[0].status.loadBalancer.ingress[0].ip}")
    api = {"embed": "v1/embeddings", "rerank": "v1/rerank" if litellm else "v1/score"}.get(m.category, "v1/chat/completions")
    if gateway_enabled():
        endpoint = f"https://{host}/{api}"
    elif m.kind == "LLMInferenceService":
        endpoint = f"https://{host}/{m.namespace}/{m.name}/{api}"
    else:
        endpoint = f"https://{host}/v2/models/{m.name}/infer"
    body = {"embed": {"model": m.name, "input": ["Hello world"]},
            "rerank": {"model": m.name, "query": "query", "documents": ["doc1", "doc2"]} if litellm
            else {"model": m.name, "text_1": "query", "text_2": ["doc1", "doc2"]}}.get(
        m.category, {"model": m.name, "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16})
    sep = "─" * 56
    print(f"\n  {CYAN}{sep}{RESET}")
    print(f"  {BOLD}Model:{RESET}    {m.name}  |  {BOLD}Kind:{RESET} {m.kind}  |  {BOLD}Category:{RESET} {m.category}")
    print(f"  {BOLD}Auth:{RESET}     {'litellm' if litellm else 'keycloak'}")
    if gw_ip:
        print(f"  {BOLD}Gateway:{RESET}  {gw_ip}  (resolve {host} → {gw_ip})")
    print(f"  {BOLD}Endpoint:{RESET} {endpoint}")
    if gateway_enabled():
        print(f'  {BOLD}Note:{RESET}     Pass "model": "{m.name}" in the request body')
    print(f"  {CYAN}{sep}{RESET}")
    if litellm:
        print(f"\n  {BOLD}1. Get your LiteLLM API key:{RESET}")
        print(f"  {DIM}export LITELLM_VIRTUAL_KEY=$(kubectl get secret {litellm_key_ref()} -n {litellm_ns()} \\")
        print(f"    -o jsonpath='{{.data.master_key}}' | base64 -d){RESET}")
    else:
        print(f"\n  {BOLD}1. Get a Keycloak token:{RESET}")
        print(f"  {DIM}source {MM_DIR}/scripts/get-keycloak-token.sh{RESET}")
        print(f"  {DIM}# exports TOKEN, GATEWAY_IP, GATEWAY_DOMAIN{RESET}")
    print(f"\n  {BOLD}2. Call the model:{RESET}")
    print(f'  {DIM}curl -sk --noproxy "*" \\')
    print(f'    -H "Authorization: Bearer {"$LITELLM_VIRTUAL_KEY" if litellm else "$TOKEN"}" '
          '-H "Content-Type: application/json" \\')
    if gw_ip:
        print(f'    --resolve "{host}:443:{gw_ip}" \\')
    print(f"    -d '{json.dumps(body, separators=(',', ':'))}' \\")
    print(f"    {endpoint}{RESET}\n")
