import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

from .k8s import K8sClient
from .ui import abort, info, warn

# Import nri_policy from scripts directory
scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
import nri_policy

def parse_cpu_quantity(cpu_val: str | int) -> int:
    """Parse a CPU specification (integer, decimal, or millicpu) and round up."""
    if isinstance(cpu_val, int):
        return cpu_val if cpu_val > 0 else 8
    val_str = str(cpu_val).strip()
    if val_str.endswith("m"):
        milli_str = val_str[:-1]
        if milli_str.isdigit():
            milli = int(milli_str)
            return (milli + 999) // 1000
    try:
        f = float(val_str)
        if f > 0:
            return math.ceil(f)
    except ValueError:
        pass
    abort(f"Invalid CPU quantity: '{cpu_val}'. Expected positive integer, decimal, or millicores (e.g. 8, 4.5, 8000m).")
    return 8

class CpuPolicyManager:
    def __init__(self, k8s: K8sClient):
        self.k8s = k8s
        self.policy: str = ""
        self._cached_policy: str = ""
        self.effective_cpu: int = 0
        self.balloon_name: str = ""
        self.annotations: dict[str, str] = {}
        self.node_selector_term: dict[str, Any] | None = None
        self._ks_full_pcpus_only: bool = False
        self._ks_threads_per_core: int = 1
        self._ks_resources_validated: bool = False

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

    def get_kubelet_config(self, node: str) -> dict[str, Any] | None:
        if not node:
            return None
        res = self.k8s.run("get", "--raw", f"/api/v1/nodes/{node}/proxy/configz")
        if res.returncode != 0 or not res.stdout:
            return None
        try:
            return json.loads(res.stdout)
        except json.JSONDecodeError:
            return None

    def is_kubelet_static(self, node: str) -> bool:
        cfg = self.get_kubelet_config(node)
        if not cfg:
            return False
        return cfg.get("kubeletconfig", {}).get("cpuManagerPolicy") == "static"

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
                info(f"Detected kubelet static CPU manager on '{probe_node}' — using kubelet-static.")
            elif self.is_nri_balloons_present():
                self.policy = "nri-balloons"
            else:
                warn("No CPU-pinning mechanism detected (no NRI balloons DaemonSet, kubelet cpuManagerPolicy is not static) — deploying best-effort, without CPU pinning.")
                self.policy = "best-effort"

        self._cached_policy = self.policy
        return self.policy

    def _kubelet_static_probe(self, node: str) -> None:
        self._ks_full_pcpus_only = False
        self._ks_threads_per_core = 1

        cfg = self.get_kubelet_config(node)
        if cfg:
            opts = cfg.get("kubeletconfig", {}).get("cpuManagerPolicyOptions", {})
            if opts.get("full-pcpus-only") == "true":
                self._ks_full_pcpus_only = True

        env_threads = os.getenv("MM_KUBELET_THREADS_PER_CORE")
        if env_threads:
            if env_threads.isdigit() and int(env_threads) > 0:
                self._ks_threads_per_core = int(env_threads)
            else:
                warn(f"Invalid MM_KUBELET_THREADS_PER_CORE='{env_threads}' — expected a positive integer; using auto-detected default.")
                if self._ks_full_pcpus_only:
                    self._ks_threads_per_core = 2
        elif self._ks_full_pcpus_only:
            self._ks_threads_per_core = 2

    def validate_resources(self, cpu: str | int, memory: str, model_name: str, node: str = "") -> int:
        parsed_cpu = parse_cpu_quantity(cpu)
        if self.policy == "kubelet-static":
            probe_node = self.probe_node(node)
            self._kubelet_static_probe(probe_node)
            effective = parsed_cpu
            tpc = self._ks_threads_per_core
            if self._ks_full_pcpus_only and tpc > 1 and effective % tpc != 0:
                effective = ((effective // tpc) + 1) * tpc

            if str(effective) != str(cpu):
                align_msg = f", full-pcpus-only aligned to {tpc}-thread cores" if self._ks_full_pcpus_only else ""
                info(f"kubelet-static: CPU {cpu} → {effective} (integer, Guaranteed QoS{align_msg})")

            if not memory:
                abort(f"kubelet-static requires an explicit memory value for '{model_name}' (Guaranteed QoS needs requests == limits). Set memory in models.yaml or pass --memory.")
            return effective

        return parsed_cpu

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
                self.node_selector_term = nri_policy.nri_node_selector_term(tp_size)
            except Exception as e:
                warn("┌─────────────────────────────────────────────────────────────────────┐")
                warn("│  NRI BALLOONS FALLBACK — Model will deploy WITHOUT CPU pinning      │")
                warn("├─────────────────────────────────────────────────────────────────────┤")
                warn(f"│  NRI policy resolution failed: {e:<52} │")
                warn("│  Action: Run the nri_cpu_balloons Ansible role, then redeploy.      │")
                warn("│  The model is being deployed in best-effort mode (no CPU pinning).  │")
                warn("└─────────────────────────────────────────────────────────────────────┘")
                annotations = {
                    "cpu-policy.model-manager.io/type": "best-effort",
                    "cpu-policy.model-manager.io/reason": "nri-resolution-failed",
                }
                effective_cpu = cpu
                self.balloon_name = ""
                self.node_selector_term = None

        elif self.policy == "kubelet-static":
            annotations["cpu-policy.model-manager.io/type"] = "kubelet-static"
            annotations["cpu-policy.model-manager.io/qos"] = "guaranteed"
            annotations["cpu-policy.model-manager.io/cpus"] = str(cpu)
            if self._ks_full_pcpus_only:
                annotations["cpu-policy.model-manager.io/full-pcpus-only"] = "true"
            if tp_size > 1:
                annotations["cpu-policy.model-manager.io/tensor-parallel"] = str(tp_size)

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
            probe = node or self.probe_node()
            if probe and not self.is_kubelet_static(probe):
                cfg = self.get_kubelet_config(probe)
                pol = cfg.get("kubeletconfig", {}).get("cpuManagerPolicy", "none") if cfg else "unknown"
                warn(f"Node '{probe}' has cpuManagerPolicy={pol} — not 'static'. {model_name} will run WITHOUT exclusive CPUs. Re-run the solutions install with kubernetes_cpu_policy=kubelet-static to reconfigure kubelet.")

            per_pod = cpu * tp
            nodes_data = self.k8s.get_json("node", node) if node else self.k8s.get_json("nodes")
            items = [nodes_data] if node and nodes_data else (nodes_data.get("items", []) if nodes_data else [])

            schedulable_nodes = []
            for n in items:
                taints = n.get("spec", {}).get("taints", [])
                if not any(t.get("effect") == "NoSchedule" for t in taints):
                    schedulable_nodes.append(n)

            if not schedulable_nodes:
                warn("Could not list nodes to verify CPU capacity — skipping the kubelet-static capacity check.")
                return

            alloc_list = []
            for n in schedulable_nodes:
                alloc_cpu = n.get("status", {}).get("allocatable", {}).get("cpu", "0")
                if str(alloc_cpu).endswith("m"):
                    alloc_cores = int(str(alloc_cpu)[:-1]) // 1000
                else:
                    alloc_cores = int(alloc_cpu) if str(alloc_cpu).isdigit() else 0
                alloc_list.append(alloc_cores)

            max_alloc = max(alloc_list) if alloc_list else 0
            total_alloc = sum(alloc_list)

            errors = []
            if max_alloc > 0 and per_pod > max_alloc:
                errors.append(f"No schedulable node has {per_pod} allocatable CPUs (cpu={cpu} x tp={tp}); the largest offers {max_alloc}. Exclusive CPUs cannot span nodes, so the pod would stay Pending.")

            if replicas > 1 and max_alloc > 0:
                want = per_pod * replicas
                if want > total_alloc:
                    warn(f"kubelet-static: {replicas} replicas x {per_pod} CPUs = {want} exceeds the cluster's {total_alloc} allocatable CPUs — some replicas will stay Pending.")

            if errors:
                warn("kubelet-static pre-flight issues detected:")
                for e in errors:
                    warn(f"  - {e}")
                if os.getenv("MM_PREFLIGHT") != "warn" and os.getenv("MM_NRI_PREFLIGHT") != "warn":
                    abort(f"kubelet-static pre-flight failed: not enough allocatable CPUs to place {model_name}.")

