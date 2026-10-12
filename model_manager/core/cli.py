import argparse
import base64
import json
import os
import re
import shutil
import subprocess  # nosec B404
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import AppConfig, ModelConfig, load_app_config
from .k8s import K8sClient
from .policy import CpuPolicyManager, parse_cpu_quantity
from .renderer import ManifestRenderer
from .routing import RoutingManager
from .ui import (
    BOLD, CYAN, DIM, GREEN, RED, RESET, YELLOW,
    abort, err, fmt_duration, info, ok, warn
)

MM_VERSION = "2.1.0"

USAGE_TEXT = f"""{BOLD}model-manager{RESET} — Deploy LLM inference models on Kubernetes

{BOLD}{CYAN}Commands:{RESET}
  {GREEN}deploy{RESET} <model>|--id <hf/repo>   Download weights + deploy serving pod
  {GREEN}cpu-collisions-check{RESET}            Report CPU pinning, NUMA layout & detect core collisions
  {GREEN}undeploy{RESET} <model>|<name>|all     Remove pod (keeps weights on PVC); accepts a catalog model, an --as alias, or any deployed name

{BOLD}{CYAN}Deploy Options:{RESET}
  {GREEN}--id{RESET} <hf/repo>       Ad-hoc deploy any HuggingFace model
  {GREEN}--name{RESET} <name>        Override generated name (with --id)
  {GREEN}--server{RESET} <name>      Model server/engine: vllm (default) | openvino  (alias: --runtime)
  {GREEN}--server-version{RESET} V   Server version (must be declared for the model if it uses servers:)
  {GREEN}--as{RESET} <name>          Deploy under a distinct name (run multiple server/version variants at once)
  {GREEN}--cpu{RESET} N              CPU cores
  {GREEN}--memory{RESET} NGi         Memory (e.g. 32Gi)
  {GREEN}--replicas{RESET} N         Replica count (default: 1)
  {GREEN}--tp{RESET} N               Tensor-parallel size (splits across N NUMA nodes)
  {GREEN}--cpu-policy{RESET} <p>     CPU pinning: nri-balloons | kubelet-static | best-effort
                            (default: MM_CPU_POLICY, else auto-detected)
  {GREEN}--category{RESET} <cat>     Override category: llm | embed | rerank | vlm
  {GREEN}--routing{RESET} <mode>     Override routing: epp | aigateway | direct
  {GREEN}--node{RESET} <name>        Pin deployment to a specific node
  {GREEN}--env{RESET} KEY=VAL        Add/override env var (repeatable)
  {GREEN}--arg{RESET} "--flag=val"   Append a runtime arg (repeatable)
  {GREEN}--label{RESET} KEY=VAL      Add a metadata label to the serving CR (repeatable);
                            use it to mark ownership — managed-by=model-manager
                            is set for every caller and cannot distinguish them
  {GREEN}--wait{RESET}               deploy: block until model is Ready; undeploy: block until pods terminate
  {GREEN}--wait-timeout{RESET} N     Seconds to wait (default: deploy 900, undeploy 180)
  {GREEN}--dry-run{RESET}            Print manifest without applying

{BOLD}{CYAN}Examples:{RESET}
  ./model-manager deploy llama3-8b-awq --wait
  ./model-manager deploy llama3-8b-awq --server-version 0.19.1 --wait
  ./model-manager deploy --id meta-llama/Llama-3.2-1B-Instruct --cpu 8 --memory 16Gi
  ./model-manager undeploy llama3-8b-awq
  ./model-manager undeploy llama3-8b-awq --wait        # block until pods terminate + CPUs are freed

{BOLD}{CYAN}For status / scale / logs, use kubectl:{RESET}
  kubectl get llminferenceservices,inferenceservices -n llm-inference
  kubectl patch llminferenceservice <name> -n llm-inference -p '{{"spec":{{"replicas":N}}}}'
  kubectl logs -n llm-inference -l app.kubernetes.io/name=<name> -f

{BOLD}{CYAN}Environment:{RESET}
  {GREEN}HF_TOKEN{RESET}              HuggingFace token for gated models
  {GREEN}MM_CONFIG{RESET}             Path to models.yaml
  {GREEN}MM_CPU_POLICY{RESET}         Cluster CPU policy: nri-balloons (default) |
                        kubelet-static | best-effort. Set by the Ansible
                        install from kubernetes_cpu_policy; --cpu-policy wins.
  {GREEN}MM_NRI_ENABLED{RESET}        Legacy: false ⇒ best-effort (no pinning)
  {GREEN}MM_PREFLIGHT{RESET}          'warn' ⇒ report placement failures, don't block
  {GREEN}MM_SKIP_PREFLIGHT{RESET}     1 ⇒ skip the placement pre-flight entirely
  {GREEN}MM_INFERENCE_HOSTNAME{RESET} Override inference gateway hostname

{BOLD}{CYAN}CPU policies:{RESET}
  {GREEN}nri-balloons{RESET}    (default) NRI balloons plugin — NUMA-aware balloons sized
                  per TP shard. Requires the nri_cpu_balloons Ansible role.
  {GREEN}kubelet-static{RESET}  kubelet static CPU manager — pods are rendered as
                  Guaranteed QoS (integer cpu, requests == limits) so kubelet
                  grants exclusive whole cores. No plugin needed. Requires the
                  cluster installed with kubernetes_cpu_policy=kubelet-static.
  {GREEN}best-effort{RESET}     No pinning; the CFS scheduler places threads."""

def require_label(pair: str) -> None:
    if "=" not in pair:
        abort(f"--label requires KEY=VALUE, got: {pair}")
    k, v = pair.split("=", 1)
    if not k:
        abort(f"--label key must not be empty: {pair}")
    label_key_re = re.compile(r"^([a-z0-9]([-a-z0-9.]*[a-z0-9])?/)?[a-zA-Z0-9]([-_.a-zA-Z0-9]*[a-zA-Z0-9])?$")
    label_val_re = re.compile(r"^[a-zA-Z0-9]([-_.a-zA-Z0-9]*[a-zA-Z0-9])?$")
    if not label_key_re.match(k):
        abort(f"--label key is not a valid Kubernetes label key: {k}")
    if v and not label_val_re.match(v):
        abort(f"--label value is not a valid Kubernetes label value: {v}")

class MMArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        if "unrecognized arguments:" in message:
            unknown = message.split("unrecognized arguments:")[-1].strip().split()[0]
            abort(f"Unknown option: {unknown}")
        abort(message)

class ModelManagerApp:
    def __init__(self, script_dir: Path):
        self.script_dir = script_dir
        self.templates_dir = Path(os.getenv("MM_TEMPLATES", script_dir / "templates"))
        self.runtimes_dir = Path(os.getenv("MM_RUNTIMES", script_dir / "runtimes"))
        self.config_file = Path(os.getenv("MM_CONFIG", script_dir / "models.yaml"))
        self.k8s = K8sClient()
        self.cpu_mgr = CpuPolicyManager(self.k8s)
        self.routing_mgr = RoutingManager(self.k8s, self.script_dir, self.templates_dir)
        self.renderer = ManifestRenderer(self.templates_dir)
        self.app_config: AppConfig | None = None

    def check_prerequisites(self) -> None:
        missing = [tool for tool in ("kubectl", "curl") if shutil.which(tool) is None]
        if missing:
            abort(f"Missing required tools: {', '.join(missing)}")

    def check_network(self) -> None:
        conn_url = self.app_config.network.connectivity_check if self.app_config else "https://huggingface.co"
        if not conn_url:
            return
        proxy = self.app_config.network.https_proxy if self.app_config else ""
        proxy_args = ["--proxy", proxy] if proxy else []
        cmd = ["curl", "-sf", "--max-time", "10"] + proxy_args + ["-o", "/dev/null", conn_url]
        res = subprocess.run(cmd, capture_output=True)  # nosec B603
        if res.returncode != 0:
            abort(f"Network check failed: cannot reach {conn_url} (proxy: {proxy or 'none'}). Check network/proxy settings in models.yaml or environment.")

    def detect_category(self, model_id: str) -> str:
        if not model_id:
            return "llm"
        proxy = self.app_config.network.https_proxy if self.app_config else ""
        proxy_args = ["--proxy", proxy] if proxy else []
        cmd = ["curl", "-sf", "--max-time", "10"] + proxy_args + [f"https://huggingface.co/api/models/{model_id}"]
        res = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603
        if res.returncode == 0:
            try:
                data = json.loads(res.stdout)
                tag = data.get("pipeline_tag", "")
                if tag in ("text-generation", "text2text-generation", "conversational"):
                    return "llm"
                if tag in ("feature-extraction", "sentence-similarity", "fill-mask"):
                    return "embed"
                if tag == "text-classification":
                    return "rerank"
                if tag in ("image-text-to-text", "visual-question-answering", "image-to-text"):
                    return "vlm"
            except json.JSONDecodeError:
                pass
        return "llm"

    def model_uses_trust_remote_code(self, model: ModelConfig, rt_cfg: dict[str, Any], extra_args: list[str]) -> bool:
        category = model.category or "llm"
        cat_info = rt_cfg.get("categories", {}).get(category, {})
        all_declared = list(cat_info.get("args", []))
        ver_cfg = rt_cfg.get("versions", {}).get(str(model.server_version), {})
        all_declared.extend(ver_cfg.get("args", []))
        all_declared.extend(model.args)
        if "--trust-remote-code" in all_declared:
            return True
        for a in extra_args:
            if a == "--trust-remote-code":
                return True
        return False

    def load_config(self) -> None:
        self.app_config = load_app_config(self.config_file)

    def resolve_model(self, target: str, args: argparse.Namespace) -> tuple[ModelConfig, dict[str, Any]]:
        self.load_config()
        if self.app_config is None:
            abort("Configuration not loaded.")

        if args.id:
            raw_name = args.name or re.sub(r"[^a-zA-Z0-9]", "-", args.id.split("/")[-1]).lower()
            model = ModelConfig(
                name=raw_name,
                model_id=args.id,
                category=args.category or "",
                namespace=self.app_config.defaults.namespace,
                cpu=args.cpu or "",
                memory=args.memory or "",
                replicas=int(args.replicas or self.app_config.defaults.replicas),
                node=args.node or "",
                routing=args.routing or "",
                runtime=args.server or self.app_config.defaults.runtime,
                server_version=args.server_version or "",
            )
        elif target and target in self.app_config.models:
            model = self.app_config.models[target]
        else:
            abort(f"Model '{target}' not found in {self.config_file}")

        # CLI overrides
        if args.as_name:
            model.name = args.as_name
        if args.cpu:
            model.cpu = args.cpu
        if args.memory:
            model.memory = args.memory
        if args.replicas:
            model.replicas = int(args.replicas)
        if args.category:
            model.category = args.category
        if args.node:
            model.node = args.node
        if args.server:
            model.runtime = args.server
        if args.server_version:
            model.server_version = args.server_version

        # Determine Runtime configuration
        runtime_name = model.runtime or model.default_server or self.app_config.defaults.runtime
        model.runtime = runtime_name
        rt_cfg = dict(self.app_config.runtimes.get(runtime_name, {}))

        # Check runtime dir values.yaml for low-level defaults (e.g. openvino/values.yaml)
        val_file = self.runtimes_dir / runtime_name / "values.yaml"
        if val_file.is_file():
            import yaml
            with open(val_file, "r", encoding="utf-8") as vf:
                v_data = yaml.safe_load(vf) or {}
                for k, v in v_data.items():
                    if k not in rt_cfg or not rt_cfg[k]:
                        rt_cfg[k] = v

        # Resolve server version
        versions_dict = rt_cfg.get("versions", {})
        if model.servers and runtime_name in model.servers:
            srv_info = model.servers[runtime_name]
            allowed_vers = srv_info.get("versions", [])
            default_v = srv_info.get("default", "")
            if not model.server_version:
                model.server_version = default_v
            elif model.server_version not in allowed_vers:
                abort(f"Server version '{model.server_version}' not allowed for model '{model.name}'. Allowed: {allowed_vers}")
        elif not model.server_version and versions_dict:
            model.server_version = str(next(iter(versions_dict.keys()), ""))

        # Category auto-detection and defaults
        routing_is_explicit = bool(args.routing or (model.routing if not args.id else False))
        if not model.category:
            model.category = self.detect_category(model.model_id)
            info(f"Category: {model.category} (auto-detected)")

        category = model.category
        cat_defaults = rt_cfg.get("categories", {}).get(category, {}).get("defaults", {})
        if not model.cpu:
            model.cpu = cat_defaults.get("cpu", 8)
        if not model.memory:
            model.memory = cat_defaults.get("memory", "16Gi")

        # Routing precedence
        if args.routing:
            model.routing = args.routing
        elif not routing_is_explicit and category in ("embed", "rerank"):
            model.routing = "direct"
        elif not model.routing:
            model.routing = rt_cfg.get("routing", "epp" if rt_cfg.get("kind") == "LLMInferenceService" else "direct")

        # Image resolution
        selected_version_cfg = versions_dict.get(str(model.server_version), {})
        if not model.image:
            if selected_version_cfg and selected_version_cfg.get("image"):
                model.image = selected_version_cfg["image"]
            elif rt_cfg.get("image"):
                model.image = rt_cfg["image"]

        model.kind = rt_cfg.get("kind", "LLMInferenceService")
        return model, rt_cfg

    def resolve_tp_size(self, opt_tp: int | None, extra_args: list[str]) -> int:
        if opt_tp:
            return opt_tp
        for i, a in enumerate(extra_args):
            if a == "--tensor-parallel-size" and i + 1 < len(extra_args):
                if extra_args[i + 1].isdigit():
                    return int(extra_args[i + 1])
            elif a.startswith("--tensor-parallel-size="):
                val = a.split("=", 1)[1]
                if val.isdigit():
                    return int(val)
        return 1

    def render_manifest(self, model: ModelConfig, rt_cfg: dict[str, Any], args: argparse.Namespace, tp_size: int = 1) -> str:
        if self.app_config is None:
            abort("Configuration not loaded.")
        category = model.category or "llm"
        cat_info = rt_cfg.get("categories", {}).get(category, {})

        # Resolve CPU policy
        self.cpu_mgr.resolve(args.cpu_policy, model.node)
        norm_cpu = self.cpu_mgr.validate_resources(model.cpu, model.memory, model.name, model.node)
        annotations, effective_cpu = self.cpu_mgr.build_annotations(norm_cpu, tp_size, model.node, model.kind)
        indent = "  " if model.kind == "LLMInferenceService" else "    "
        annotations_block = self.renderer.build_annotations_block(annotations, indent=indent)

        # Build Env
        merged_env: dict[str, str] = {}
        # 0. Trust-remote-code implied offline serving
        if self.model_uses_trust_remote_code(model, rt_cfg, args.arg or []):
            merged_env["HF_HUB_OFFLINE"] = "1"
        # 1. Base runtime env
        merged_env.update(rt_cfg.get("env", {}))
        # 2. Category env
        cat_env = cat_info.get("env", {})
        kv_cache_gb = str(cat_info.get("defaults", {}).get("kv_cache_gb", 8))
        for k, v in cat_env.items():
            merged_env[k] = str(v).replace("${KV_CACHE_GB}", kv_cache_gb)
        # 2.5 Version env delta
        ver_cfg = rt_cfg.get("versions", {}).get(str(model.server_version), {})
        merged_env.update(ver_cfg.get("env", {}))
        # 3. Model env
        merged_env.update(model.env)
        # 4. CLI extra env
        for e in args.env or []:
            if "=" in e:
                k, v = e.split("=", 1)
                merged_env[k] = v

        hf_token = os.getenv("HF_TOKEN", "")
        env_block = self.renderer.build_env_block(
            merged_env, model.runtime,
            self.app_config.network.http_proxy,
            self.app_config.network.https_proxy,
            self.app_config.network.no_proxy,
            hf_token
        )

        # Build Args
        all_args: list[str] = []
        all_args.extend(cat_info.get("args", []))
        # Version args delta if any
        ver_cfg = rt_cfg.get("versions", {}).get(str(model.server_version), {})
        all_args.extend(ver_cfg.get("args", []))
        all_args.extend(model.args)
        if model.kind == "LLMInferenceService":
            all_args.extend(["--served-model-name", model.name])
            if model.chat_template:
                if model.chat_template.startswith("/"):
                    all_args.extend(["--chat-template", model.chat_template])
                else:
                    all_args.extend(["--chat-template", "/etc/chat-template/chat_template.jinja"])
            if tp_size > 1 and "--tensor-parallel-size" not in " ".join(all_args):
                all_args.extend(["--tensor-parallel-size", str(tp_size)])

        if args.arg:
            all_args.extend(args.arg)
        args_block = self.renderer.build_args_block(all_args)

        # Labels
        labels_dict = dict(model.extra_labels)
        for lbl in args.label or []:
            if "=" in lbl:
                k, v = lbl.split("=", 1)
                labels_dict[k] = v
        labels_block = self.renderer.build_labels_block(labels_dict)

        # Affinity
        require_amx = self.app_config.defaults.require_amx
        affinity_block = self.renderer.build_affinity_block(model.node, require_amx, self.cpu_mgr.node_selector_term)

        # Chat template mount/volume
        chat_mount, chat_volume = "", ""
        if model.chat_template and not model.chat_template.startswith("/"):
            chat_mount = (
                "        volumeMounts:\n"
                "          - name: chat-template\n"
                "            mountPath: /etc/chat-template\n"
                "            readOnly: true\n"
            )
            chat_volume = (
                "    volumes:\n"
                "      - name: chat-template\n"
                "        configMap:\n"
                f"          name: {model.name}-chat-template\n"
            )

        model_uri = f"pvc://{self.app_config.storage.pvc_name}/{model.name}"

        if model.kind == "LLMInferenceService":
            return self.renderer._render("llm-inference-service.yaml", {
                "MODEL_NAME": model.name,
                "NAMESPACE": model.namespace,
                "IMAGE": model.image,
                "CPU": str(effective_cpu),
                "MEMORY": model.memory,
                "REPLICAS": str(model.replicas),
                "MODEL_URI": model_uri,
                "ARGS": args_block,
                "ENV": env_block,
                "NRI_ANNOTATIONS": annotations_block,
                "GATEWAY_NAME": self.routing_mgr.ai_gateway_name,
                "GATEWAY_NAMESPACE": self.routing_mgr.ai_gateway_ns,
                "CHAT_TEMPLATE_MOUNT": chat_mount,
                "CHAT_TEMPLATE_VOLUME": chat_volume,
                "AFFINITY": affinity_block,
                "EXTRA_LABELS": labels_block
            })
        else:
            tmpl = "inference-service-ovms.yaml" if model.runtime == "openvino" else "inference-service.yaml"
            runtime_name = rt_cfg.get("serving_runtime_name", f"{model.runtime}-runtime")
            return self.renderer._render(tmpl, {
                "MODEL_NAME": model.name,
                "NAMESPACE": model.namespace,
                "MODEL_ID": model.model_id,
                "MODEL_FORMAT": rt_cfg.get("format", "openvino"),
                "MODEL_URI": model_uri,
                "PVC_NAME": self.app_config.storage.pvc_name,
                "RUNTIME_NAME": runtime_name,
                "CPU": str(effective_cpu),
                "MEMORY": model.memory,
                "REPLICAS": str(model.replicas),
                "ARGS": args_block,
                "ENV": env_block,
                "AFFINITY": affinity_block,
                "NRI_ANNOTATIONS": annotations_block,
                "EXTRA_LABELS": labels_block
            })

    def print_endpoints(self, model: ModelConfig) -> None:
        host = self.routing_mgr.resolve_inference_hostname()
        gw_ip = self.routing_mgr.gateway_ip()
        endpoint = self.routing_mgr.build_endpoint_url(model.name, model.kind, model.category, model.namespace)
        auth = self.routing_mgr.auth_provider()
        unified = self.routing_mgr.aigateway_enabled()

        token_hint = "$LITELLM_VIRTUAL_KEY" if auth == "litellm" else "$TOKEN"
        sep = "─" * 56

        print(f"\n  {CYAN}{sep}{RESET}")
        print(f"  {BOLD}Model:{RESET}    {model.name}  |  {BOLD}Kind:{RESET} {model.kind}  |  {BOLD}Category:{RESET} {model.category}")
        print(f"  {BOLD}Auth:{RESET}     {auth}")
        if gw_ip:
            print(f"  {BOLD}Gateway:{RESET}  {gw_ip}  (resolve {host} → {gw_ip})")
        print(f"  {BOLD}Endpoint:{RESET} {endpoint}")
        if unified:
            print(f'  {BOLD}Note:{RESET}     Pass "model": "{model.name}" in the request body')
        print(f"  {CYAN}{sep}{RESET}")

        if auth == "litellm":
            res_name = self.routing_mgr.litellm_key_ref
            res_ns = self.routing_mgr.litellm_ns
            print(f"\n  {BOLD}1. Get your LiteLLM API key:{RESET}")
            print(f"  {DIM}export LITELLM_VIRTUAL_KEY=$(kubectl get secret {res_name} -n {res_ns} \\")
            print(f"    -o jsonpath='{{.data.master_key}}' | base64 -d){RESET}")
        else:
            print(f"\n  {BOLD}1. Get a Keycloak token:{RESET}")
            print(f"  {DIM}source {self.script_dir}/scripts/get-keycloak-token.sh{RESET}")
            print(f"  {DIM}# exports TOKEN, GATEWAY_IP, GATEWAY_DOMAIN{RESET}")

        if model.category == "embed":
            sample_body = json.dumps({"model": model.name, "input": ["Hello world"]})
        elif model.category == "rerank":
            sample_body = json.dumps({"model": model.name, "text_1": "query", "text_2": ["doc1", "doc2"]})
        else:
            sample_body = json.dumps({"model": model.name, "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16})

        print(f"\n  {BOLD}2. Call the model:{RESET}")
        print(f"  {DIM}curl -sk --noproxy \"*\" \\")
        print(f"    -H \"Authorization: Bearer {token_hint}\" -H \"Content-Type: application/json\" \\")
        if gw_ip:
            print(f"    --resolve \"{host}:443:{gw_ip}\" \\")
        print(f"    -d '{sample_body}' \\")
        print(f"    {endpoint}{RESET}\n")

    def workload_ready(self, kind: str, namespace: str, name: str) -> str:
        raw = self.k8s.get_json(kind, name, "-n", namespace)
        if not raw:
            return "NotFound"
        conditions = raw.get("status", {}).get("conditions", [])
        for c in conditions:
            if c.get("type") == "MainWorkloadReady":
                return str(c.get("status", ""))
        for c in conditions:
            if c.get("type") == "Ready":
                return str(c.get("status", ""))
        return "Pending"

    def on_model_ready(self, model: ModelConfig) -> None:
        ok(f"{model.name} is Ready")
        if model.kind == "LLMInferenceService":
            self.routing_mgr.patch_pool_status(model.name, model.namespace)
        internal_url = self.routing_mgr.model_internal_url(model.name, model.namespace, model.kind)
        self.routing_mgr.litellm_register(model.name, internal_url)

    def on_model_ready_by_name(self, model_name: str, namespace: str, kind: str) -> None:
        ok(f"{model_name} is Ready")
        if kind == "LLMInferenceService":
            self.routing_mgr.patch_pool_status(model_name, namespace)
        internal_url = self.routing_mgr.model_internal_url(model_name, namespace, kind)
        self.routing_mgr.litellm_register(model_name, internal_url)

    def pod_progress(self, namespace: str, model_name: str) -> str:
        pods = self.k8s.get_json("pods", "-n", namespace, "-l", f"app.kubernetes.io/name={model_name}")
        if not pods or not pods.get("items"):
            return "waiting for pod to be created"
        pod = pods["items"][0]
        status = pod.get("status", {})
        phase = status.get("phase", "unknown")
        c_statuses = status.get("containerStatuses", [])
        ready = False
        wait_reason = ""
        if c_statuses:
            ready = bool(c_statuses[0].get("ready", False))
            waiting = c_statuses[0].get("state", {}).get("waiting", {})
            wait_reason = waiting.get("reason", "")

        if phase == "Pending":
            if wait_reason == "ContainerCreating":
                return "creating container (pulling image / mounting weights)"
            if wait_reason in ("ImagePullBackOff", "ErrImagePull"):
                return f"image pull failed ({wait_reason})"
            if wait_reason == "CrashLoopBackOff":
                return f"container crashing ({wait_reason}) — check logs"
            return f"pending ({wait_reason})" if wait_reason else "pending (waiting to be scheduled)"
        elif phase == "Running":
            return "running — finalizing routes" if ready else "running — loading model / warming up"
        elif phase == "Failed":
            return "pod failed — check logs"
        elif phase == "Succeeded":
            return "pod exited"
        return phase

    def ensure_runtime_installed(self, rt_cfg: dict[str, Any], runtime_name: str) -> None:
        cr_name = rt_cfg.get("serving_runtime_name") or f"{runtime_name}-runtime"
        if not self.k8s.get_raw("clusterservingruntime", cr_name):
            manifest_ptr = rt_cfg.get("serving_runtime_manifest")
            rt_file = None
            if manifest_ptr and (self.runtimes_dir / manifest_ptr).is_file():
                rt_file = self.runtimes_dir / manifest_ptr
            elif (self.runtimes_dir / runtime_name / "serving-runtime.yaml").is_file():
                rt_file = self.runtimes_dir / runtime_name / "serving-runtime.yaml"
            elif (self.runtimes_dir / runtime_name / "runtime.yaml").is_file():
                rt_file = self.runtimes_dir / runtime_name / "runtime.yaml"

            if rt_file and rt_file.is_file():
                info(f"Installing ClusterServingRuntime: {cr_name}")
                content = rt_file.read_text(encoding="utf-8")
                content = content.replace("{{RUNTIME_IMAGE}}", rt_cfg.get("image", ""))
                self.k8s.apply(content)

    def download_weights(self, model: ModelConfig, rt_cfg: dict[str, Any], extra_args: list[str]) -> None:
        self.check_network()

        pvc_name = self.app_config.storage.pvc_name
        hf_token = os.getenv("HF_TOKEN", "")

        # Refuse to download a gated model without an HF_TOKEN
        if not hf_token:
            proxy = self.app_config.network.https_proxy if self.app_config else ""
            proxy_args = ["--proxy", proxy] if proxy else []
            cmd = ["curl", "-sf", "--max-time", "10"] + proxy_args + [f"https://huggingface.co/api/models/{model.model_id}"]
            res = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603
            if res.returncode == 0:
                try:
                    data = json.loads(res.stdout)
                    gated = data.get("gated")
                    if gated and gated != False and str(gated).lower() != "false":
                        abort(f"Model '{model.model_id}' is gated. Set HF_TOKEN with an authorized token.")
                except json.JSONDecodeError:
                    pass

        info(f"Downloading model weights: {model.model_id} → PVC:/{model.name}")

        if hf_token:
            res = self.k8s.run(
                "create", "secret", "generic", "hf-token",
                "-n", model.namespace,
                f"--from-literal=token={hf_token}",
                "--dry-run=client", "-o", "yaml"
            )
            if res.returncode == 0:
                self.k8s.apply(res.stdout)

        downloader_image = rt_cfg.get("downloader_image") or model.image
        is_ov = (model.runtime == "openvino")
        job_uid, job_gid = ("5000", "5000") if is_ov else ("1000", "1000")

        sentinel = f"/models/{model.name}/.download_complete"
        if is_ov:
            stage = f"/models/.pull-{model.name}"
            dest = f"/models/{model.name}"
            cat_info = rt_cfg.get("categories", {}).get(model.category or "llm", {})
            ov_wf = cat_info.get("weight_format", "")
            ov_wf_flag = f" --weight-format {ov_wf}" if ov_wf else ""
            target_device = os.getenv("MM_OVMS_TARGET_DEVICE", "CPU")
            dl_cmd = (
                "set -e\n"
                "export HOME=/models HF_HOME=/models/.hf-cache\n"
                f"if [ -f {sentinel} ]; then echo 'Model already prepared, skipping'; exit 0; fi\n"
                f"rm -rf '{stage}'; mkdir -p '{stage}'\n"
                f"ovms --pull --source_model '{model.model_id}' --model_repository_path '{stage}' "
                f"--model_name '{model.name}' --task text_generation{ov_wf_flag} --target_device {target_device}\n"
                f"graph_dir=$(dirname $(find '{stage}' -name graph.pbtxt | head -1))\n"
                "if [ -z \"$graph_dir\" ]; then echo 'ERROR: pull produced no graph.pbtxt'; exit 1; fi\n"
                f"rm -rf '{dest}'; mkdir -p '{dest}'\n"
                f"mv \"$graph_dir\"/* '{dest}'/\n"
                f"rm -rf '{stage}'\n"
                f"printf '%s\\n' '{{ \"mediapipe_config_list\": [ {{ \"name\": \"{model.name}\", \"base_path\": \".\" }} ], \"model_config_list\": [] }}' > '{dest}/config_all.json'\n"
                f"touch {sentinel}\n"
                f"echo 'Prepared OVMS servable + config_all.json at {dest}'"
            )
        else:
            trc_prefetch = ""
            if self.model_uses_trust_remote_code(model, rt_cfg, extra_args):
                trc_prefetch = (
                    f"\ncfg=/models/{model.name}/config.json\n"
                    "if [ -f \"$cfg\" ]; then\n"
                    "  echo 'trust-remote-code: prefetching custom modeling code'\n"
                    "  for repo in $(python3 -c \"import json; c=json.load(open('$cfg')); print(chr(10).join({v.split('--')[0] for v in (c.get('auto_map') or {}).values() if '--' in v}))\"); do\n"
                    f"    echo \"  fetching $repo\"; hf download \"$repo\" --local-dir /models/{model.name} --include '*.py'\n"
                    "  done\n"
                    "  python3 -c \"import json; p='$cfg'; c=json.load(open(p)); am=c.get('auto_map') or {}; c['auto_map']={k:(v.split('--')[-1] if '--' in v else v) for k,v in am.items()}; json.dump(c,open(p,'w'),indent=2)\"\n"
                    "  echo '  patched auto_map for offline load'\n"
                    "fi"
                )
            dl_cmd = (
                "set -e\n"
                "export HOME=/tmp HF_HOME=/tmp/.hf XDG_CACHE_HOME=/tmp/.cache PIP_CACHE_DIR=/tmp/.pip\n"
                f"if [ -f {sentinel} ]; then echo 'Weights already present, skipping download'; exit 0; fi\n"
                "command -v hf >/dev/null 2>&1 || pip install -q huggingface_hub[hf_xet]\n"
                f"hf download '{model.model_id}' --local-dir /models/{model.name}{trc_prefetch}\n"
                f"touch {sentinel}"
            )

        indented_cmd = "\n".join("              " + line for line in dl_cmd.splitlines())
        env_lines = [
            "            - name: HOME\n              value: \"/models\"",
            "            - name: HF_HOME\n              value: \"/models/.hf-cache\""
        ]
        if hf_token:
            env_lines.append("            - name: HF_TOKEN\n              valueFrom:\n                secretKeyRef:\n                  name: hf-token\n                  key: token")
        for k, v in [("HTTP_PROXY", self.app_config.network.http_proxy),
                     ("HTTPS_PROXY", self.app_config.network.https_proxy),
                     ("NO_PROXY", self.app_config.network.no_proxy)]:
            if v:
                env_lines.append(f"            - name: {k}\n              value: \"{v}\"")
                env_lines.append(f"            - name: {k.lower()}\n              value: \"{v}\"")

        env_block = "          env:\n" + "\n".join(env_lines)

        job_yaml = self.renderer._render("download-job.yaml", {
            "MODEL_NAME": model.name,
            "NAMESPACE": model.namespace,
            "MODEL_ID": model.model_id,
            "PVC_NAME": pvc_name,
            "DOWNLOADER_IMAGE": downloader_image,
            "ENV_BLOCK": env_block,
            "DOWNLOAD_CMD": indented_cmd,
            "JOB_UID": job_uid,
            "JOB_GID": job_gid,
            "DL_CPU_REQUEST": "500m",
            "DL_MEM_REQUEST": "2Gi",
            "DL_CPU_LIMIT": str(model.cpu or 2),
            "DL_MEM_LIMIT": str(model.memory or "8Gi"),
            "TTL_SECONDS": "300",
            "TIMEOUT": "7200"
        })

        self.k8s.delete("job", f"download-{model.name}", model.namespace)
        self.k8s.apply(job_yaml)
        info(f"Downloading model weights: {model.model_id}")
        info(f"  {YELLOW}This may take several minutes depending on model size and internet speed.{RESET}")
        info("  Follow progress below — do not interrupt.")
        time.sleep(3)

        # Stream logs in background
        log_proc = None
        for _ in range(10):
            pod_name = self.k8s.get_raw("pods", "-n", model.namespace, "-l", f"job-name=download-{model.name}", "-o", "jsonpath={.items[0].metadata.name}")
            if pod_name:
                try:
                    cmd = self.k8s._cmd("logs", "-f", pod_name, "-n", model.namespace)
                    log_proc = subprocess.Popen(cmd)  # nosec B603
                except OSError as e:
                    warn(f"Failed to stream download job logs: {e}")
                break
            time.sleep(1)

        # Wait for completion
        res = self.k8s.run("wait", "--for=condition=complete", f"job/download-{model.name}",
                           "-n", model.namespace, "--timeout=7200s")
        if log_proc and log_proc.poll() is None:
            try:
                log_proc.terminate()
            except OSError as e:
                warn(f"Could not terminate log streamer: {e}")

        if res.returncode == 0:
            ok(f"Download complete: {model.name}")
        else:
            logs = self.k8s.get_raw("logs", "-l", f"job-name=download-{model.name}", "-n", model.namespace, "--tail=20")
            low_logs = logs.lower()
            if "401" in logs or "access denied" in low_logs or "gated" in low_logs or "authorization" in low_logs:
                abort(f"Download failed: model '{model.model_id}' requires authentication. Set HF_TOKEN with an authorized token.")
            elif "connecterror" in low_logs or "connectionerror" in low_logs or "timeout" in low_logs:
                abort("Download failed: network error reaching HuggingFace. Check proxy settings.")
            abort(f"Download job failed. Check logs: kubectl logs -l job-name=download-{model.name} -n {model.namespace}")

    def cmd_deploy(self, target: str, args: argparse.Namespace) -> None:
        model, rt_cfg = self.resolve_model(target, args)

        info(f"Deploying {model.name} ({model.kind})")
        info(f"  Runtime: {model.runtime}{f' @ {model.server_version}' if model.server_version else ''} | Category: {model.category} | Routing: {model.routing or 'default'}")
        info(f"  CPU: {model.cpu} | Memory: {model.memory} | Replicas: {model.replicas}")
        if model.node:
            info(f"  Node: {model.node}")
        self.cpu_mgr.resolve(args.cpu_policy, model.node)
        info(f"  CPU policy: {self.cpu_mgr.policy}")

        tp_size = self.resolve_tp_size(args.tp, args.arg or [])
        manifest = self.render_manifest(model, rt_cfg, args, tp_size)
        if args.dry_run:
            print(manifest)
            return

        self.k8s.ensure_cluster_reachable()

        # Diff check
        action = "apply"
        force_redeploy = os.getenv("MM_FORCE_REDEPLOY", "").lower() in ("true", "1", "yes")
        existing = self.k8s.get_raw(model.kind, model.name, "-n", model.namespace)
        if existing and not force_redeploy:
            diff_rc = self.k8s.diff_ssa(manifest)
            if diff_rc == 0:
                action = "skip"
                ok(f"{model.name} already deployed, spec unchanged — skipping apply")
            elif diff_rc == 1:
                info(f"{model.name} spec changed — re-applying")
            else:
                if self.workload_ready(model.kind, model.namespace, model.name) == "True":
                    action = "skip"
                    warn(f"{model.name} spec diff failed (rc={diff_rc}) but workload is Ready — skipping apply")
                else:
                    warn(f"{model.name} spec diff failed (rc={diff_rc}) — applying")

        if model.chat_template and not model.chat_template.startswith("/"):
            self.k8s.ensure_namespace(model.namespace)
            cm_body = json.dumps({
                "apiVersion": "v1", "kind": "ConfigMap",
                "metadata": {"name": f"{model.name}-chat-template", "namespace": model.namespace},
                "data": {"chat_template.jinja": model.chat_template}
            })
            info(f"Storing chat template in ConfigMap: {model.name}-chat-template")
            self.k8s.apply(cm_body)

        if action != "skip":
            parsed_cpu = parse_cpu_quantity(model.cpu)
            self.cpu_mgr.preflight_gate(parsed_cpu, tp_size, model.node, model.replicas, model.name)
            self.k8s.ensure_storage(model.namespace, self.app_config.storage.pvc_name)
            self.download_weights(model, rt_cfg, args.arg or [])
            self.ensure_runtime_installed(rt_cfg, model.runtime)
            self.k8s.apply_ssa(manifest)
            ok(f"Manifest applied: {model.kind}/{model.name} in {model.namespace}")

        self.routing_mgr.aigateway_register(model.name, model.namespace, model.kind, model.routing, model.category)

        if args.wait:
            info(f"Waiting for {model.name} to become Ready (timeout: {args.wait_timeout}s)...")
            start = time.time()
            end = start + args.wait_timeout
            last_beat = 0
            while time.time() < end:
                if self.workload_ready(model.kind, model.namespace, model.name) == "True":
                    self.on_model_ready(model)
                    self.print_endpoints(model)
                    return
                if time.time() - last_beat >= 15:
                    elapsed = int(time.time() - start)
                    msg = self.pod_progress(model.namespace, model.name)
                    info(f"  {DIM}[{fmt_duration(elapsed)}] {model.name}: {msg}{RESET}")
                    last_beat = time.time()
                time.sleep(5)
            warn(f"{model.name} did not become Ready within {args.wait_timeout}s")
            warn(f"  Check progress: kubectl logs -n {model.namespace} -l app.kubernetes.io/name={model.name} -f")
        else:
            time.sleep(3)
            if self.workload_ready(model.kind, model.namespace, model.name) == "True":
                self.on_model_ready(model)
            else:
                recon_dir = Path(os.getenv("MM_RECONCILE_LOG_DIR", os.getenv("TMPDIR", tempfile.gettempdir())))
                recon_log = recon_dir / f"model-manager-reconcile-{model.name}.log"
                recon_cmd = [
                    sys.executable, "-m", "core", "_reconcile",
                    model.name, model.namespace, model.kind, str(args.wait_timeout)
                ]
                try:
                    with open(recon_log, "w", encoding="utf-8") as rf:
                        subprocess.Popen(recon_cmd, stdout=rf, stderr=rf, start_new_session=True)  # nosec B603
                except OSError as e:
                    warn(f"Failed to spawn detached reconciler: {e}")

                info(f"{model.name} is starting in the background.")
                info(f"  {YELLOW}Model load + warmup can take a few minutes — try inference after a short while.{RESET}")
                if self.routing_mgr.auth_provider() == "litellm":
                    info("  Registration with LiteLLM/gateway completes automatically once it is Ready.")
                else:
                    info("  AI Gateway routing is reconciled automatically once it is Ready.")
                info(f"  Check status: kubectl get {model.kind} {model.name} -n {model.namespace}")
                info(f"  Follow logs:  kubectl logs -n {model.namespace} -l app.kubernetes.io/name={model.name} -f")
                info(f"  Reconcile log: {recon_log}")
                info("  (or re-run with --wait to block until it is Ready)")
        self.print_endpoints(model)

    def _resolve_deployed(self, target: str) -> tuple[str, str, str] | None:
        ns = self.app_config.defaults.namespace if self.app_config else "llm-inference"
        for kind in ("LLMInferenceService", "InferenceService"):
            found_ns = self.k8s.get_raw(kind, target, "-n", ns, "-o", "jsonpath={.metadata.namespace}")
            if not found_ns:
                found_ns = self.k8s.get_raw(kind, "-A", "--field-selector", f"metadata.name={target}", "-o", "jsonpath={.items[0].metadata.namespace}")
            if found_ns:
                if self.k8s.get_raw("httproute", f"direct-{target}", "-n", found_ns):
                    routing = "direct"
                elif self.k8s.get_raw("aigatewayroute", f"aigw-{target}", "-n", found_ns) and self.k8s.get_raw("aiservicebackend", target, "-n", found_ns):
                    routing = "aigateway"
                else:
                    routing = "epp"
                info(f"Discovered live {kind}/{target} in {found_ns} (no catalog entry)")
                return kind, found_ns, routing
        return None

    def cmd_undeploy(self, target: str, args: argparse.Namespace) -> None:
        if not target:
            abort("No model specified. Use: model-manager undeploy <name>|all")

        self.k8s.ensure_cluster_reachable()
        self.load_config()
        if self.app_config is None:
            abort("Configuration not loaded.")

        if target == "all":
            for name in list(self.app_config.models.keys()):
                self.cmd_undeploy(name, args)
            return

        ns = self.app_config.defaults.namespace
        kind = "LLMInferenceService"
        routing = "epp"
        if target in self.app_config.models:
            m = self.app_config.models[target]
            kind, ns, routing = m.kind or kind, m.namespace or ns, m.routing or routing
        else:
            resolved = self._resolve_deployed(target)
            if not resolved:
                abort(f"No catalog entry or deployed resource named '{target}' found.")
            kind, ns, routing = resolved

        info(f"Undeploying {target}")
        self.k8s.delete(kind, target, ns)
        self.k8s.delete("configmap", f"{target}-chat-template", ns)
        self.routing_mgr.aigateway_deregister(target, ns, kind, routing)
        self.routing_mgr.litellm_deregister(target)

        if args.wait:
            timeout = args.wait_timeout or 180
            info(f"Waiting for {target} pods to terminate (timeout: {timeout}s)...")
            start = time.time()
            end = start + timeout
            last_beat = 0
            while time.time() < end:
                pods = self.k8s.get_raw("pods", "-n", ns, "-l", f"app.kubernetes.io/name={target}", "-o", "name")
                if not pods.strip():
                    ok(f"{target} undeployed (all pods terminated, CPUs released)")
                    return
                if time.time() - last_beat >= 15:
                    elapsed = int(time.time() - start)
                    count = len(pods.strip().splitlines())
                    info(f"  {DIM}[{fmt_duration(elapsed)}] waiting for {count} pod(s) to terminate...{RESET}")
                    last_beat = time.time()
                time.sleep(3)
            warn(f"{target} undeployed, but pods are still terminating after {timeout}s — CPUs/balloons may not be reclaimed yet")
            warn(f"  Check: kubectl get pods -n {ns} -l app.kubernetes.io/name={target}")
        else:
            ok(f"{target} undeployed")

    def cmd_reconcile(self, model_name: str, namespace: str, kind: str, timeout: int) -> None:
        start = time.time()
        end = start + timeout
        while time.time() < end:
            if self.workload_ready(kind, namespace, model_name) == "True":
                self.on_model_ready_by_name(model_name, namespace, kind)
                sys.exit(0)
            time.sleep(10)
        warn(f"{model_name} not Ready within {timeout}s — gave up before reconciling routing.")
        warn(f"  Reconcile with: ./model-manager deploy {model_name}")
        sys.exit(1)

    def cmd_cpu_collisions_check(self) -> None:
        self.load_config()
        if self.app_config is None:
            abort("Configuration not loaded.")
        ns = self.app_config.defaults.namespace
        report_script = self.script_dir / "scripts" / "nri-cpu-report.py"
        if not report_script.is_file():
            abort(f"nri-cpu-report.py not found at {report_script}")
        info(f"Running CPU pinning & collision report (namespace: {ns})")
        py_exe = sys.executable
        cmd = [py_exe, str(report_script), "--namespace", ns]
        if self.k8s.kubeconfig:
            cmd.extend(["--kubeconfig", self.k8s.kubeconfig])
        env = os.environ.copy()
        for p in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            env.pop(p, None)
        subprocess.run(cmd, env=env, check=False)  # nosec B603

def build_parser() -> MMArgumentParser:
    parser = MMArgumentParser(
        prog="model-manager",
        description="model-manager — Deploy LLM inference models on Kubernetes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="store_true", help="show this help message and exit")
    parser.add_argument("-v", "--version", action="store_true", help="show program's version number and exit")

    subparsers = parser.add_subparsers(dest="command")

    # Deploy
    deploy_p = subparsers.add_parser("deploy", help="Download weights + deploy serving pod", add_help=False)
    deploy_p.add_argument("model", nargs="?", default="", help="Catalog model name")
    deploy_p.add_argument("--id", help="Ad-hoc deploy any HuggingFace model")
    deploy_p.add_argument("--name", help="Override generated name (with --id)")
    deploy_p.add_argument("--server", "--runtime", dest="server", help="Model server/engine: vllm | openvino")
    deploy_p.add_argument("--server-version", help="Server version")
    deploy_p.add_argument("--as", dest="as_name", help="Deploy under a distinct name")
    deploy_p.add_argument("--cpu", help="CPU cores")
    deploy_p.add_argument("--memory", help="Memory (e.g. 32Gi)")
    deploy_p.add_argument("--replicas", type=int, help="Replica count")
    deploy_p.add_argument("--tp", type=int, help="Tensor-parallel size")
    deploy_p.add_argument("--cpu-policy", help="CPU pinning: nri-balloons | kubelet-static | best-effort")
    deploy_p.add_argument("--category", help="Override category: llm | embed | rerank | vlm")
    deploy_p.add_argument("--routing", help="Override routing: epp | aigateway | direct")
    deploy_p.add_argument("--node", help="Pin deployment to a specific node")
    deploy_p.add_argument("--env", action="append", help="Add/override env var (repeatable)")
    deploy_p.add_argument("--arg", action="append", help="Append runtime arg (repeatable)")
    deploy_p.add_argument("--label", action="append", help="Add metadata label KEY=VAL (repeatable)")
    deploy_p.add_argument("--wait", action="store_true", help="Block until model is Ready")
    deploy_p.add_argument("--wait-timeout", type=int, default=900, help="Seconds to wait (default: 900)")
    deploy_p.add_argument("--dry-run", action="store_true", help="Print manifest without applying")
    deploy_p.add_argument("-h", "--help", action="store_true")

    # Undeploy
    undep_p = subparsers.add_parser("undeploy", help="Remove pod (keeps weights on PVC)", add_help=False)
    undep_p.add_argument("model", nargs="?", default="", help="Catalog model, alias, or 'all'")
    undep_p.add_argument("--wait", action="store_true", help="Block until pods terminate")
    undep_p.add_argument("--wait-timeout", type=int, default=180, help="Seconds to wait (default: 180)")
    undep_p.add_argument("-h", "--help", action="store_true")

    # CPU collisions
    cpu_p = subparsers.add_parser("cpu-collisions-check", help="Report CPU pinning & collisions", add_help=False)
    cpu_p.add_argument("-h", "--help", action="store_true")

    # Internal reconcile worker
    recon_p = subparsers.add_parser("_reconcile", add_help=False)
    recon_p.add_argument("recon_model")
    recon_p.add_argument("recon_ns")
    recon_p.add_argument("recon_kind")
    recon_p.add_argument("recon_timeout", type=int)

    # help and version subcommands
    subparsers.add_parser("help", add_help=False)
    subparsers.add_parser("version", add_help=False)

    return parser

def main() -> None:
    raw_args = sys.argv[1:]
    if not raw_args:
        print(USAGE_TEXT)
        sys.exit(1)

    if raw_args[0] in ("version", "-v", "--version"):
        print(f"model-manager {MM_VERSION}")
        sys.exit(0)

    if raw_args[0] in ("help", "-h", "--help"):
        print(USAGE_TEXT)
        sys.exit(0)

    # Rearrange options if they appear before the subcommand (e.g. `./model-manager --wait deploy qwen3-0-6b`)
    known_cmds = {"deploy", "undeploy", "cpu-collisions-check", "version", "help", "_reconcile"}
    cmd_idx = -1
    for idx, arg in enumerate(raw_args):
        if arg in known_cmds:
            cmd_idx = idx
            break

    if cmd_idx > 0:
        raw_args = [raw_args[cmd_idx]] + raw_args[:cmd_idx] + raw_args[cmd_idx + 1:]
    elif cmd_idx == -1:
        if any(a in ("-h", "--help") for a in raw_args):
            print(USAGE_TEXT)
            sys.exit(0)
        if any(a in ("-v", "--version") for a in raw_args):
            print(f"model-manager {MM_VERSION}")
            sys.exit(0)
        if raw_args[0].startswith("-"):
            abort(f"Unknown option: {raw_args[0]}")
        abort(f"Unknown command: {raw_args[0]} (run ./model-manager --help)")

    parser = build_parser()
    args = parser.parse_args(raw_args)

    if getattr(args, "help", False) or args.command == "help":
        print(USAGE_TEXT)
        sys.exit(0)

    if getattr(args, "version", False) or args.command == "version":
        print(f"model-manager {MM_VERSION}")
        sys.exit(0)

    # Validate labels
    if getattr(args, "label", None):
        for lbl in args.label:
            require_label(lbl)

    # Lowercase cpu-policy
    if getattr(args, "cpu_policy", None) and args.cpu_policy:
        args.cpu_policy = args.cpu_policy.lower()

    script_dir = Path(__file__).resolve().parent.parent
    app = ModelManagerApp(script_dir)

    if args.command == "_reconcile":
        app.cmd_reconcile(args.recon_model, args.recon_ns, args.recon_kind, args.recon_timeout)
        sys.exit(0)

    app.check_prerequisites()

    if args.command == "deploy":
        if not args.model and not args.id:
            abort("No model specified. Use: model-manager deploy <model-name> or --id <hf/repo>")
        app.cmd_deploy(args.model, args)
    elif args.command == "undeploy":
        app.cmd_undeploy(args.model, args)
    elif args.command == "cpu-collisions-check":
        app.cmd_cpu_collisions_check()
    else:
        print(USAGE_TEXT)
        sys.exit(1)

if __name__ == "__main__":
    main()

