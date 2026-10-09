import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from mm.config import discover_global_config, discover_inference_config
from mm.k8s import K8sClient
from mm.ui import abort, info, ok, warn

class RoutingManager:
    def __init__(self, k8s: K8sClient, script_dir: Path, templates_dir: Path):
        self.k8s = k8s
        self.script_dir = script_dir
        self.templates_dir = templates_dir
        self.ai_gateway_name = os.getenv("MM_AI_GATEWAY_NAME", "ai-gateway")
        self.ai_gateway_ns = os.getenv("MM_AI_GATEWAY_NS", "llm-inference")
        self.route_timeout = os.getenv("MM_AIGATEWAY_ROUTE_TIMEOUT", "600s")
        self.litellm_ns = os.getenv("MM_LITELLM_NAMESPACE", "litellm")
        self.litellm_svc = os.getenv("MM_LITELLM_SERVICE", "litellm")
        self.litellm_secret = os.getenv("MM_LITELLM_SECRET", "litellm-master-key")

    def auth_provider(self) -> str:
        env_val = os.getenv("MM_LITELLM_ENABLED")
        if env_val:
            if env_val.lower() in ("true", "1", "yes"):
                return "litellm"
            if env_val.lower() in ("false", "0", "no"):
                return "keycloak"

        gc = discover_global_config(self.script_dir)
        if gc and gc.is_file():
            try:
                import yaml
                with open(gc, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                if data.get("auth_provider") == "litellm":
                    return "litellm"
            except Exception:
                pass
        return "keycloak"

    def ai_gateway_provider(self) -> str:
        env_val = os.getenv("MM_AI_GATEWAY_PROVIDER")
        if env_val in ("agentgateway", "envoy"):
            return env_val

        for path_finder in (discover_inference_config, discover_global_config):
            cfg = path_finder(self.script_dir)
            if cfg and cfg.is_file():
                try:
                    import yaml
                    with open(cfg, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                    p = data.get("ai_gateway_provider")
                    if p in ("agentgateway", "envoy"):
                        return p
                except Exception:
                    pass
        return "envoy"

    def aigateway_enabled(self) -> bool:
        val = os.getenv("MM_AIGATEWAY_ENABLED", "")
        return val.lower() not in ("false", "0", "no")

    def effective_routing(self, routing: str, kind: str) -> str:
        if routing:
            return routing
        return "epp" if kind == "LLMInferenceService" else "direct"

    def gateway_ip(self) -> str:
        ns = os.getenv("MM_EG_GATEWAY_NS", "envoy-gateway-system")
        gw = os.getenv("MM_EG_GATEWAY_NAME", "eg-gateway")
        res = self.k8s.get_json("svc", "-n", ns, "-l", f"gateway.envoyproxy.io/owning-gateway-name={gw}")
        if res and res.get("items"):
            try:
                return res["items"][0]["status"]["loadBalancer"]["ingress"][0]["ip"]
            except (KeyError, IndexError):
                pass
        return ""

    def resolve_inference_hostname(self) -> str:
        if os.getenv("MM_INFERENCE_HOSTNAME"):
            return os.environ["MM_INFERENCE_HOSTNAME"]

        gc = discover_global_config(self.script_dir)
        gc_data = {}
        if gc and gc.is_file():
            try:
                import yaml
                with open(gc, "r", encoding="utf-8") as f:
                    gc_data = yaml.safe_load(f) or {}
            except Exception:
                pass

        prefix = ""
        if self.auth_provider() == "litellm":
            ic = discover_inference_config(self.script_dir)
            for c_path in (ic, gc):
                if c_path and c_path.is_file():
                    try:
                        import yaml
                        with open(c_path, "r", encoding="utf-8") as f:
                            d = yaml.safe_load(f) or {}
                        if d.get("litellm_hostname"):
                            return d["litellm_hostname"]
                    except Exception:
                        pass
            prefix = "litellm"

        if gc_data:
            if not prefix:
                if gc_data.get("inference_hostname"):
                    return gc_data["inference_hostname"]
                prefix = gc_data.get("inference_prefix", "inference")
            domain = gc_data.get("gateway_domain") or gc_data.get("base_domain_name", "")
            if domain:
                return f"{prefix}.{domain}"

        warn("Could not resolve inference hostname — set MM_INFERENCE_HOSTNAME or MM_GLOBAL_CONFIG")
        return ""

    def build_endpoint_url(self, model_name: str, model_kind: str, category: str) -> str:
        host = self.resolve_inference_hostname()
        if category == "embed":
            api_path = "v1/embeddings"
        elif category == "rerank":
            api_path = "v1/score"
        else:
            api_path = "v1/chat/completions"

        if self.aigateway_enabled():
            return f"https://{host}/{api_path}"
        if model_kind == "LLMInferenceService":
            return f"https://{host}/{self.ai_gateway_ns}/{model_name}/{api_path}"
        return f"https://{host}/v2/models/{model_name}/infer"

    def model_internal_url(self, model_name: str, namespace: str, model_kind: str) -> str:
        api_version = "v1"
        if os.getenv("MM_LITELLM_KSERVE_URL"):
            base = os.environ["MM_LITELLM_KSERVE_URL"].rstrip("/")
            return f"{base}/{namespace}/{model_name}/{api_version}"

        if model_kind == "InferenceService":
            res = self.k8s.get_json("svc", f"{model_name}-predictor", "-n", namespace)
            port = 80
            if res and res.get("spec", {}).get("ports"):
                port = res["spec"]["ports"][0]["port"]
            return f"http://{model_name}-predictor.{namespace}.svc.cluster.local:{port}/{api_version}"

        gw_ns = os.getenv("MM_INFERENCE_GATEWAY_NS", "envoy-gateway-system")
        gw_name = os.getenv("MM_INFERENCE_GATEWAY", "ai-gateway")
        svc = self.k8s.get_raw("svc", gw_name, "-n", gw_ns, "-o", "jsonpath={.metadata.name}")
        if svc:
            return f"http://{svc}.{gw_ns}.svc.cluster.local/{namespace}/{model_name}/{api_version}"

        warn(f"AI Gateway service '{gw_name}' not found in {gw_ns} — LiteLLM will bypass gateway for {model_name}")
        return f"http://{model_name}-kserve-workload-svc.{namespace}.svc.cluster.local:8000/{api_version}"

    def patch_pool_status(self, model_name: str, namespace: str) -> None:
        pool = f"{model_name}-inference-pool"
        for _ in range(24):
            if self.k8s.run("get", "inferencepool", pool, "-n", namespace).returncode == 0:
                break
            time.sleep(5)
        if self.k8s.run("get", "inferencepool", pool, "-n", namespace).returncode != 0:
            return

        gw_name = self.k8s.get_raw("llminferenceservice", model_name, "-n", namespace,
                                   "-o", "jsonpath={.spec.router.gateway.refs[0].name}") or os.getenv("MM_INFERENCE_GATEWAY", "ai-gateway")
        gw_ns = self.k8s.get_raw("llminferenceservice", model_name, "-n", namespace,
                                 "-o", "jsonpath={.spec.router.gateway.refs[0].namespace}") or os.getenv("MM_INFERENCE_GATEWAY_NS", "llm-inference")

        controller = "gateway.envoyproxy.io/gatewayclass-controller"
        if self.ai_gateway_provider() == "agentgateway":
            controller = "agentgateway.dev/agentgateway"

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        patch_body = json.dumps({
            "status": {
                "parents": [{
                    "parentRef": {
                        "group": "gateway.networking.k8s.io",
                        "kind": "Gateway",
                        "name": gw_name,
                        "namespace": gw_ns
                    },
                    "controllerName": controller,
                    "conditions": [{
                        "type": "Accepted",
                        "status": "True",
                        "reason": "Accepted",
                        "message": "Patched by model-manager",
                        "lastTransitionTime": now
                    }]
                }]
            }
        })
        self.k8s.run("patch", "inferencepool", pool, "-n", namespace,
                     "--type=merge", "--subresource=status", "-p", patch_body)

    def _render_template(self, tmpl_name: str, vars_dict: dict[str, str]) -> str:
        tmpl_file = self.templates_dir / tmpl_name
        if not tmpl_file.is_file():
            abort(f"Template not found: {tmpl_file}")
        content = tmpl_file.read_text(encoding="utf-8")
        for k, v in vars_dict.items():
            content = content.replace(f"{{{{{k}}}}}", v)
        return content

    def aigateway_register(self, model_name: str, namespace: str, kind: str, routing: str, category: str) -> None:
        if not self.aigateway_enabled():
            return
        eff_routing = self.effective_routing(routing, kind)
        provider = self.ai_gateway_provider()

        if eff_routing == "epp":
            pool_name = f"{model_name}-inference-pool"
            if provider == "agentgateway":
                tmpl = "agw-route-pool.yaml"
                info(f"Creating HTTPRoute aigw-{model_name} (agentgateway → InferencePool {pool_name})")
            else:
                tmpl = "aigateway-route.yaml"
                info(f"Creating AIGatewayRoute aigw-{model_name} (EPP → {pool_name})")
            manifest = self._render_template(tmpl, {
                "MODEL_NAME": model_name,
                "NAMESPACE": namespace,
                "POOL_NAME": pool_name,
                "ROUTE_TIMEOUT": self.route_timeout,
                "AI_GATEWAY_NAME": self.ai_gateway_name,
                "AI_GATEWAY_NAMESPACE": self.ai_gateway_ns
            })
            self.k8s.apply_ssa(manifest)

        elif eff_routing == "aigateway":
            if kind == "InferenceService":
                svc_name, svc_port = f"{model_name}-predictor", "80"
            else:
                svc_name, svc_port = f"{model_name}-kserve-workload-svc", "8000"

            if provider == "agentgateway":
                tmpl = "agw-route-direct-isvc.yaml" if kind == "InferenceService" else "agw-route-direct-llm.yaml"
                info(f"Creating HTTPRoute direct-{model_name} (agentgateway direct → {svc_name}:{svc_port})")
            else:
                tmpl = "aigateway-route-direct.yaml"
                info(f"Creating AIGatewayRoute aigw-{model_name} (aigateway → {svc_name}:{svc_port})")

            manifest = self._render_template(tmpl, {
                "MODEL_NAME": model_name,
                "NAMESPACE": namespace,
                "SERVICE_NAME": svc_name,
                "SERVICE_PORT": svc_port,
                "ROUTE_TIMEOUT": self.route_timeout,
                "AI_GATEWAY_NAME": self.ai_gateway_name,
                "AI_GATEWAY_NAMESPACE": self.ai_gateway_ns
            })
            self.k8s.apply_ssa(manifest)

        else: # direct
            if kind == "InferenceService":
                svc_name, svc_port = f"{model_name}-predictor", "80"
                tmpl = "httproute-direct-isvc.yaml"
            else:
                svc_name, svc_port = f"{model_name}-kserve-workload-svc", "8000"
                if category == "embed":
                    tmpl = "httproute-direct-embed.yaml"
                elif category == "rerank":
                    tmpl = "httproute-direct-rerank.yaml"
                else:
                    tmpl = "httproute-direct-llm.yaml"

            info(f"Creating HTTPRoute direct-{model_name} (direct → {svc_name}:{svc_port}, {category or 'llm'})")
            manifest = self._render_template(tmpl, {
                "MODEL_NAME": model_name,
                "NAMESPACE": namespace,
                "SERVICE_NAME": svc_name,
                "SERVICE_PORT": svc_port,
                "ROUTE_TIMEOUT": self.route_timeout,
                "AI_GATEWAY_NAME": self.ai_gateway_name,
                "AI_GATEWAY_NAMESPACE": self.ai_gateway_ns
            })
            self.k8s.apply_ssa(manifest)

    def aigateway_deregister(self, model_name: str, namespace: str, kind: str, routing: str) -> None:
        if not self.aigateway_enabled():
            return
        eff_routing = self.effective_routing(routing, kind)
        provider = self.ai_gateway_provider()

        if eff_routing == "epp":
            cr = "httproute" if provider == "agentgateway" else "aigatewayroute"
            self.k8s.delete(cr, f"aigw-{model_name}", namespace)
        elif eff_routing == "aigateway":
            if provider == "agentgateway":
                self.k8s.delete("httproute", f"direct-{model_name}", namespace)
            else:
                self.k8s.delete("aigatewayroute", f"aigw-{model_name}", namespace)
                self.k8s.delete("aiservicebackend", model_name, namespace)
                self.k8s.delete("backend", model_name, namespace)
        else:
            self.k8s.delete("httproute", f"direct-{model_name}", namespace)

    # ── LiteLLM support ──

    def litellm_url(self) -> str:
        if os.getenv("MM_LITELLM_URL"):
            return os.environ["MM_LITELLM_URL"].rstrip("/")
        port = self.k8s.get_raw("svc", self.litellm_svc, "-n", self.litellm_ns, "-o", "jsonpath={.spec.ports[0].port}")
        if port:
            return f"http://{self.litellm_svc}.{self.litellm_ns}.svc.cluster.local:{port}"
        return ""

    def litellm_key(self) -> str:
        if os.getenv("MM_LITELLM_MASTER_KEY"):
            return os.environ["MM_LITELLM_MASTER_KEY"]
        enc = self.k8s.get_raw("secret", self.litellm_secret, "-n", self.litellm_ns, "-o", "jsonpath={.data.master_key}")
        if enc:
            try:
                return base64.b64decode(enc).decode("utf-8")
            except Exception:
                pass
        return ""

    def litellm_curl(self, method: str, url: str, data: str | None = None) -> str:
        key = self.litellm_key()
        curl_args = f"-sf -X {method} -H 'Content-Type: application/json'"
        if key:
            curl_args += f" -H 'Authorization: Bearer {key}'"
        if data:
            curl_args += f" -d '{data}'"
        curl_args += f" '{url}'"
        pod_name = f"mm-curl-{int(time.time() * 1000) % 100000}"
        res = self.k8s.run("run", pod_name, "--rm", "-i", "--restart=Never",
                           f"--image={os.getenv('MM_CURL_IMAGE', 'curlimages/curl:8.12.1')}",
                           "-n", self.litellm_ns, "--", "sh", "-c", f"curl {curl_args}")
        out = res.stdout.strip()
        idx = out.rfind("}")
        return out[:idx+1] if idx != -1 else out

    def litellm_model_ids(self, model_name: str, base_url: str) -> list[str]:
        raw = self.litellm_curl("GET", f"{base_url}/model/info")
        if not raw:
            return []
        try:
            data = json.loads(raw)
            return [
                m["model_info"]["id"]
                for m in data.get("data", [])
                if m.get("model_name") == model_name and m.get("model_info", {}).get("id")
            ]
        except Exception:
            return []

    def litellm_register(self, model_name: str, model_url: str) -> None:
        if self.auth_provider() != "litellm":
            return
        base = self.litellm_url()
        if not base:
            warn("LiteLLM not reachable — skipping registration")
            return

        self.litellm_deregister(model_name)
        payload = json.dumps({
            "model_name": model_name,
            "litellm_params": {
                "model": f"openai/{model_name}",
                "api_base": model_url,
                "api_key": "unused"
            }
        })
        info(f"Registering {model_name} with LiteLLM (api_base={model_url})")
        for attempt in range(1, 4):
            self.litellm_curl("POST", f"{base}/model/new", payload)
            if self.litellm_model_ids(model_name, base):
                ok("Registered with LiteLLM")
                return
            if attempt < 3:
                warn(f"LiteLLM registration attempt {attempt} failed — retrying")
                time.sleep(attempt * 5)
        warn(f"Could not register {model_name} with LiteLLM after 3 attempts.")

    def litellm_deregister(self, model_name: str) -> None:
        if self.auth_provider() != "litellm":
            return
        base = self.litellm_url()
        if not base:
            return
        ids = self.litellm_model_ids(model_name, base)
        if not ids:
            self.litellm_curl("POST", f"{base}/model/delete", json.dumps({"id": model_name}))
            return
        for mid in ids:
            self.litellm_curl("POST", f"{base}/model/delete", json.dumps({"id": mid}))
