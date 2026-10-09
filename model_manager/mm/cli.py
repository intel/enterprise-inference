import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from mm.config import AppConfig, ModelConfig, load_app_config
from mm.k8s import K8sClient
from mm.policy import CpuPolicyManager
from mm.renderer import ManifestRenderer
from mm.routing import RoutingManager
from mm.ui import (
    BOLD, CYAN, DIM, GREEN, RED, RESET, YELLOW,
    abort, err, fmt_duration, info, ok, warn
)

MM_VERSION = "2.1.0"

class ModelManagerApp:
    def __init__(self, script_dir: Path):
        self.script_dir = script_dir
        self.templates_dir = script_dir / "templates"
        self.runtimes_dir = script_dir / "runtimes"
        self.config_file = Path(os.getenv("MM_CONFIG", script_dir / "models.yaml"))
        self.k8s = K8sClient()
        self.cpu_mgr = CpuPolicyManager(self.k8s)
        self.routing_mgr = RoutingManager(self.k8s, self.script_dir, self.templates_dir)
        self.renderer = ManifestRenderer(self.templates_dir)
        self.app_config: AppConfig | None = None

    def check_prerequisites(self) -> None:
        missing = []
        for tool in ("kubectl", "curl"):
            res = os.system(f"command -v {tool} >/dev/null 2>&1")
            if res != 0:
                missing.append(tool)
        if missing:
            abort(f"Missing required tools: {', '.join(missing)}")

    def load_config(self) -> None:
        self.app_config = load_app_config(self.config_file)

    def resolve_model(self, target: str, args: argparse.Namespace) -> tuple[ModelConfig, dict[str, Any]]:
        self.load_config()
        assert self.app_config is not None

        if args.id:
            raw_name = args.name or re.sub(r"[^a-zA-Z0-9]", "-", args.id.split("/")[-1]).lower()
            model = ModelConfig(
                name=raw_name,
                model_id=args.id,
                category=args.category or "llm",
                namespace=self.app_config.defaults.namespace,
                cpu=args.cpu or 8,
                memory=args.memory or "16Gi",
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

        # Check legacy/runtime dir values.yaml for low-level defaults (e.g. openvino/values.yaml)
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

        # Category defaults
        category = model.category or "llm"
        cat_defaults = rt_cfg.get("categories", {}).get(category, {}).get("defaults", {})
        if not model.cpu:
            model.cpu = cat_defaults.get("cpu", 8)
        if not model.memory:
            model.memory = cat_defaults.get("memory", "16Gi")

        # Routing precedence
        if args.routing:
            model.routing = args.routing
        elif not model.routing:
            if category in ("embed", "rerank"):
                model.routing = "direct"
            else:
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

    def render_manifest(self, model: ModelConfig, rt_cfg: dict[str, Any], args: argparse.Namespace) -> str:
        assert self.app_config is not None
        category = model.category or "llm"
        cat_info = rt_cfg.get("categories", {}).get(category, {})

        # Compute TP
        tp_size = 1
        if args.tp:
            tp_size = int(args.tp)
        else:
            for i, a in enumerate(args.arg or []):
                if a == "--tensor-parallel-size" and i + 1 < len(args.arg):
                    tp_size = int(args.arg[i + 1])
                elif a.startswith("--tensor-parallel-size="):
                    tp_size = int(a.split("=")[1])

        # Resolve CPU policy
        self.cpu_mgr.resolve(args.cpu_policy, model.node)
        norm_cpu = self.cpu_mgr.validate_resources(model.cpu, model.memory, model.node)
        annotations, effective_cpu = self.cpu_mgr.build_annotations(norm_cpu, tp_size, model.node, model.kind)
        indent = "  " if model.kind == "LLMInferenceService" else "    "
        annotations_block = self.renderer.build_annotations_block(annotations, indent=indent)

        # Build Env
        merged_env: dict[str, str] = {}
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
        affinity_block = self.renderer.build_affinity_block(model.node, require_amx)

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
        endpoint = self.routing_mgr.build_endpoint_url(model.name, model.kind, model.category)
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
            print(f"\n  {BOLD}1. Get your LiteLLM API key:{RESET}")
            print(f"  {DIM}export LITELLM_VIRTUAL_KEY=$(kubectl get secret {self.routing_mgr.litellm_secret} -n {self.routing_mgr.litellm_ns} \\")
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

    def download_weights(self, model: ModelConfig, rt_cfg: dict[str, Any]) -> None:
        pvc_name = self.app_config.storage.pvc_name
        info(f"Downloading model weights: {model.model_id} → PVC:/{model.name}")

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            sec_yaml = (
                "apiVersion: v1\nkind: Secret\nmetadata:\n"
                f"  name: hf-token\n  namespace: {model.namespace}\n"
                "type: Opaque\nstringData:\n"
                f"  token: {hf_token}\n"
            )
            self.k8s.apply(sec_yaml)

        downloader_image = rt_cfg.get("downloader_image") or model.image
        is_ov = (model.runtime == "openvino")
        job_uid, job_gid = ("5000", "5000") if is_ov else ("1000", "1000")

        sentinel = f"/models/{model.name}/.download_complete"
        if is_ov:
            stage = f"/models/.pull-{model.name}"
            dest = f"/models/{model.name}"
            dl_cmd = (
                "set -e\n"
                "export HOME=/models HF_HOME=/models/.hf-cache\n"
                f"if [ -f {sentinel} ]; then echo 'Model already prepared, skipping'; exit 0; fi\n"
                f"rm -rf '{stage}'; mkdir -p '{stage}'\n"
                f"ovms --pull --source_model '{model.model_id}' --model_repository_path '{stage}' "
                f"--model_name '{model.name}' --task text_generation --target_device CPU\n"
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
            dl_cmd = (
                "set -e\n"
                "export HOME=/tmp HF_HOME=/tmp/.hf XDG_CACHE_HOME=/tmp/.cache PIP_CACHE_DIR=/tmp/.pip\n"
                f"if [ -f {sentinel} ]; then echo 'Weights already present, skipping download'; exit 0; fi\n"
                "command -v hf >/dev/null 2>&1 || pip install -q huggingface_hub[hf_xet]\n"
                f"hf download '{model.model_id}' --local-dir /models/{model.name}\n"
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
                    import subprocess
                    cmd = self.k8s._cmd("logs", "-f", pod_name, "-n", model.namespace)
                    log_proc = subprocess.Popen(cmd)
                except Exception:
                    pass
                break
            time.sleep(1)

        # Wait for completion
        res = self.k8s.run("wait", "--for=condition=complete", f"job/download-{model.name}",
                           "-n", model.namespace, "--timeout=7200s")
        if log_proc:
            try:
                log_proc.terminate()
            except Exception:
                pass

        if res.returncode == 0:
            ok(f"Download complete: {model.name}")
        else:
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

        manifest = self.render_manifest(model, rt_cfg, args)
        if args.dry_run:
            print(manifest)
            return

        # Diff check
        action = "apply"
        existing = self.k8s.get_raw(model.kind, model.name, "-n", model.namespace)
        if existing and not os.getenv("MM_FORCE_REDEPLOY"):
            diff_rc = self.k8s.diff_ssa(manifest)
            if diff_rc == 0:
                action = "skip"
                ok(f"{model.name} already deployed, spec unchanged — skipping apply")
            elif diff_rc == 1:
                info(f"{model.name} spec changed — re-applying")

        if model.chat_template and not model.chat_template.startswith("/"):
            cm_body = json.dumps({
                "apiVersion": "v1", "kind": "ConfigMap",
                "metadata": {"name": f"{model.name}-chat-template", "namespace": model.namespace},
                "data": {"chat_template.jinja": model.chat_template}
            })
            info(f"Storing chat template in ConfigMap: {model.name}-chat-template")
            self.k8s.apply(cm_body)

        if action != "skip":
            self.cpu_mgr.preflight_gate(int(model.cpu or 8), 1, model.node, model.replicas, model.name)
            self.k8s.ensure_storage(model.namespace, self.app_config.storage.pvc_name)
            self.download_weights(model, rt_cfg)
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
        else:
            time.sleep(3)
            if self.workload_ready(model.kind, model.namespace, model.name) == "True":
                self.on_model_ready(model)
            else:
                info(f"{model.name} is starting in the background.")
                info(f"  {YELLOW}Model load + warmup can take a few minutes — try inference after a short while.{RESET}")
        self.print_endpoints(model)

    def cmd_undeploy(self, target: str, args: argparse.Namespace) -> None:
        if not target:
            abort("No model specified. Use: model-manager undeploy <name>|all")

        self.load_config()
        assert self.app_config is not None

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
            # Discover live
            for k in ("LLMInferenceService", "InferenceService"):
                found_ns = self.k8s.get_raw(k, target, "-o", "jsonpath={.metadata.namespace}")
                if found_ns:
                    kind = k
                    ns = found_ns
                    break

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
            warn(f"{target} undeployed, but pods are still terminating after {timeout}s")
        else:
            ok(f"{target} undeployed")

    def cmd_cpu_collisions_check(self) -> None:
        self.load_config()
        assert self.app_config is not None
        ns = self.app_config.defaults.namespace
        report_script = self.script_dir / "scripts" / "nri-cpu-report.py"
        if not report_script.is_file():
            abort(f"nri-cpu-report.py not found at {report_script}")
        info(f"Running CPU pinning & collision report (namespace: {ns})")
        py_exe = sys.executable
        k_flag = f"--kubeconfig {self.k8s.kubeconfig}" if self.k8s.kubeconfig else ""
        env_cmd = "env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy"
        os.system(f"{env_cmd} {py_exe} {report_script} --namespace {ns} {k_flag}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-manager",
        description="model-manager — Deploy LLM inference models on Kubernetes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--version", action="version", version=f"model-manager {MM_VERSION}")

    subparsers = parser.add_subparsers(dest="command")

    # Deploy
    deploy_p = subparsers.add_parser("deploy", help="Download weights + deploy serving pod")
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

    # Undeploy
    undep_p = subparsers.add_parser("undeploy", help="Remove pod (keeps weights on PVC)")
    undep_p.add_argument("model", help="Catalog model, alias, or 'all'")
    undep_p.add_argument("--wait", action="store_true", help="Block until pods terminate")
    undep_p.add_argument("--wait-timeout", type=int, default=180, help="Seconds to wait (default: 180)")

    # CPU collisions
    subparsers.add_parser("cpu-collisions-check", help="Report CPU pinning & collisions")

    return parser


def main() -> None:
    parser = build_parser()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent.parent
    app = ModelManagerApp(script_dir)
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
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
