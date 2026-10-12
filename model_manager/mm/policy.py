import json
import os
import re
from typing import Any

from mm.k8s import K8sClient
from mm.ui import abort, info, warn

# Import nri_policy from scripts
try:
    import nri_policy
except ImportError:
    try:
        from scripts import nri_policy
    except ImportError:
        from mm import nri_policy

class CpuPolicyManager:
    def __init__(self, k8s: K8sClient):
        self.k8s = k8s
        self.policy: str = ""
        self._cached_policy: str = ""
        self.effective_cpu: int = 0
        self.balloon_name: str = ""
        self.annotations: dict[str, str] = {}
        self.node_selector_term: dict[str, Any] | None = None

    def probe_node(self, node_pref: str = "") -> str:
        if node_pref:
            return node_pref
        nodes = self.k8s.get_json("nodes", "-l", "!node-role.kubernetes.io/control-plane")
        if nodes and nodes.get("items"):
            return nodes["items"][0]["metadata"]["name"]
        nodes = self.k8s.get_json("nodes")
        if nodes and nodes.get("items"):
            return nodes["items"][0]["metadata"]["name"]
        return ""

    def is_kubelet_static(self, node: str) -> bool:
        if not node:
            return False
        res = self.k8s.run("get", "--raw", f"/api/v1/nodes/{node}/proxy/configz")
        if res.returncode != 0 or not res.stdout:
            return False
        try:
            cfg = json.loads(res.stdout)
            return cfg.get("kubeletconfig", {}).get("cpuManagerPolicy") == "static"
        except Exception:
            return False

    def is_nri_balloons_present(self) -> bool:
        res = self.k8s.run("get", "daemonset", "nri-resource-policy-balloons", "-n", "kube-system", "-o", "name")
        return res.returncode == 0 and bool(res.stdout.strip())

    def resolve(self, opt_policy: str = "", node_pref: str = "") -> str:
        if self._cached_policy:
            self.policy = self._cached_policy
            return self.policy

        requested = ""
        if opt_policy:
            requested = opt_policy.lower()
            if requested not in ("nri-balloons", "kubelet-static", "best-effort"):
                abort(f"Invalid --cpu-policy '{opt_policy}'. Valid: nri-balloons, kubelet-static, best-effort.")
        elif os.getenv("MM_CPU_POLICY"):
            env_pol = os.environ["MM_CPU_POLICY"].lower()
            if env_pol in ("nri-balloons", "kubelet-static", "best-effort"):
                requested = env_pol
            else:
                warn(f"Invalid MM_CPU_POLICY='{env_pol}' — ignoring, falling back to auto-detection.")

        if not requested and os.getenv("MM_NRI_ENABLED", "").lower() in ("false", "0", "no"):
            requested = "best-effort"

        if requested:
            self.policy = requested
        else:
            # Auto-detect
            probe_node = self.probe_node(node_pref)
            if self.is_kubelet_static(probe_node):
                self.policy = "kubelet-static"
            elif self.is_nri_balloons_present():
                self.policy = "nri-balloons"
            else:
                warn("No CPU-pinning mechanism detected (no NRI balloons DaemonSet, kubelet cpuManagerPolicy is not static) — deploying best-effort, without CPU pinning.")
                self.policy = "best-effort"

        self._cached_policy = self.policy
        return self.policy

    def validate_resources(self, cpu: str | int, memory: str, node: str = "") -> int:
        cpu_int = int(cpu) if str(cpu).isdigit() else 8
        if self.policy == "kubelet-static":
            if not str(cpu).isdigit():
                warn(f"kubelet-static requires integer CPU requests; rounding '{cpu}' → {cpu_int}")
            # check full-pcpus-only if possible
            probe_node = self.probe_node(node)
            res = self.k8s.run("get", "--raw", f"/api/v1/nodes/{probe_node}/proxy/configz")
            if res.returncode == 0:
                try:
                    cfg = json.loads(res.stdout)
                    opts = cfg.get("kubeletconfig", {}).get("cpuManagerPolicyOptions", {})
                    if opts.get("full-pcpus-only") == "true" and cpu_int % 2 != 0:
                        abort(f"kubelet-static on node '{probe_node}' has full-pcpus-only=true enabled: CPU request must be a multiple of whole physical cores (even number), got cpu={cpu_int}.")
                except Exception:
                    pass
        elif self.policy == "nri-balloons":
            pass
        return cpu_int

    def build_annotations(self, cpu: int, tp_size: int, node: str, kind: str) -> tuple[dict[str, str], int]:
        annotations: dict[str, str] = {
            "cpu-policy.model-manager.io/type": self.policy
        }
        effective_cpu = cpu

        if self.policy == "nri-balloons":
            try:
                b_annos = nri_policy.build_annotations(tp_size, kind)
                annotations.update(b_annos)
                effective_cpu = nri_policy.effective_cpu_request(cpu, node)
                self.effective_cpu = effective_cpu
                self.balloon_name = b_annos.get("cpu-policy.model-manager.io/balloon", "")
            except Exception as e:
                warn(f"Failed to resolve NRI balloons annotations: {e}")

        return annotations, effective_cpu

    def preflight_gate(self, cpu: int, tp: int, node: str, replicas: int, model_name: str) -> None:
        if os.getenv("MM_SKIP_PREFLIGHT") == "1" or os.getenv("MM_NRI_SKIP_PREFLIGHT") == "1":
            return

        if self.policy == "nri-balloons":
            if not os.getenv("MM_NRI_ENABLED", "true").lower() in ("true", "1", "yes"):
                return
            result = nri_policy.preflight(cpu, tp, node=node or "", replicas=replicas)
            if not result.get("ok", False):
                errors = result.get("errors", [])
                err_msg = "\n".join(errors)
                if os.getenv("MM_PREFLIGHT") == "warn" or os.getenv("MM_NRI_PREFLIGHT") == "warn":
                    warn(f"NRI pre-flight check reported warnings:\n{err_msg}")
                    return
                abort(f"NRI pre-flight failed: not enough physical cores to place {model_name} (cpu={cpu} tp={tp} replicas={replicas}) without spilling onto reserved hyperthread-sibling cores.\n{err_msg}")
        elif self.policy == "kubelet-static":
            pass
