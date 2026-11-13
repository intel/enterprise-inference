"""CPU pinning policies. Precedence: --cpu-policy > MM_CPU_POLICY > MM_NRI_ENABLED=false > cluster auto-detect."""
from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import math
import os
import re

from . import k8s, nri_policy
from .config import Model, is_uint, pick, text, truthy
from .ui import abort, info, warn

ANN = "cpu-policy.model-manager.io"
# Where a deployed serving CR keeps its container's CPU request.
LIVE_CPU = {"LLMInferenceService": "{.spec.template.containers[0].resources.requests.cpu}",
            "InferenceService": "{.spec.predictor.model.resources.requests.cpu}"}
NRI_FALLBACK_BOX = """\
┌─────────────────────────────────────────────────────────────────────┐
│  NRI BALLOONS FALLBACK — Model will deploy WITHOUT CPU pinning      │
├─────────────────────────────────────────────────────────────────────┤
│  NRI policy resolution failed. Possible causes:                     │
│    • nri-resource-policy-balloons DaemonSet not deployed             │
│    • NRI ConfigMap missing or misconfigured                          │
│    • kubectl connectivity issue to kube-system                       │
│                                                                     │
│  Action: Run the nri_cpu_balloons Ansible role, then redeploy.      │
│  The model is being deployed in best-effort mode (no CPU pinning).  │
└─────────────────────────────────────────────────────────────────────┘"""


def probe_node(node: str = "") -> str:
    """The pinned node, else the first worker, else the first node."""
    jsonpath = "jsonpath={.items[0].metadata.name}"
    return (node or k8s.out("get", "nodes", "-l", "!node-role.kubernetes.io/control-plane", "-o", jsonpath)
            or k8s.out("get", "nodes", "-o", jsonpath))


def kubelet_config(node: str) -> dict | None:
    """The node's live kubelet config (needs RBAC on nodes/proxy); None when unreadable."""
    try:
        cfg = json.loads(k8s.out("get", "--raw", f"/api/v1/nodes/{node}/proxy/configz") or "null")
    except ValueError:
        return None
    return cfg.get("kubeletconfig") if isinstance(cfg, dict) else None


class BestEffort:
    """No pinning; the CFS scheduler places threads."""
    name = label = "best-effort"
    counts_running_pods = False  # the pre-flight counts free capacity, which a running replica already holds

    def prepare(self, m: Model) -> None:
        """Normalise m.cpu/m.memory for the policy; may abort."""

    def annotations(self, m: Model, tp: int) -> tuple[dict, str]:
        """Pod annotations and the CPU quantity to request."""
        return {f"{ANN}/type": self.name}, m.cpu

    def preflight(self, m: Model, tp: int) -> list[str]:
        """Reasons the model cannot be placed (empty when it fits)."""
        return []

    def failure(self, m: Model, tp: int) -> str:
        return (f"Pre-flight failed: cannot place {m.name} (cpu={m.cpu} memory={m.memory} replicas={m.replicas}) "
                "on the nodes it can use. Lower --cpu/--memory/--replicas/--max-replicas, undeploy a model, or add a "
                "node. Override with MM_PREFLIGHT=warn (proceed anyway) or MM_SKIP_PREFLIGHT=1 (skip check).")


class NriBalloons(BestEffort):
    """NRI balloons plugin: NUMA-aware balloons sized per tensor-parallel shard."""
    name = "nri-balloons"
    counts_running_pods = True
    label = "NRI"

    def annotations(self, m: Model, tp: int) -> tuple[dict, str]:
        if not is_uint(m.cpu):
            for line in NRI_FALLBACK_BOX.splitlines():
                warn(line)
            return {f"{ANN}/type": "best-effort", f"{ANN}/reason": "nri-resolution-failed"}, m.cpu
        nri_policy.report_daemonset()
        anns = nri_policy.build_annotations(tp, m.kind)
        cpu = nri_policy.effective_cpu_request(int(m.cpu))
        if cpu > 0 and str(cpu) != m.cpu:
            info(f"NRI: Effective CPU {m.cpu} → {cpu} (hideHyperthreads active)", err=True)
        return anns, str(cpu) if cpu > 0 else m.cpu

    def _errors(self, m: Model, tp: int) -> tuple[bool, list[str]]:
        """(the gate passes, placement errors); with MM_NRI_PREFLIGHT=warn the gate passes despite errors."""
        if not truthy(os.environ.get("MM_NRI_ENABLED", "true")):
            return True, []
        if not is_uint(m.cpu):
            return False, [f"nri-balloons needs a whole number of CPUs, got cpu={m.cpu}"]
        with contextlib.redirect_stderr(io.StringIO()):  # notes are noise here; errors are reported by the caller
            result = nri_policy.preflight(int(m.cpu), tp, node=m.node, replicas=int(m.replicas))
        return result["ok"], result["errors"]

    def preflight(self, m: Model, tp: int) -> list[str]:
        ok, errors = self._errors(m, tp)
        return [] if ok else errors


    def failure(self, m: Model, tp: int) -> str:
        return (f"NRI pre-flight failed: not enough physical cores to place {m.name} (cpu={m.cpu} tp={tp} "
                f"replicas={m.replicas}) without spilling onto reserved hyperthread-sibling cores. Lower "
                "--cpu/--replicas, raise --tp, undeploy a model, or add a node. Override with MM_NRI_PREFLIGHT=warn "
                "(proceed anyway) or MM_NRI_SKIP_PREFLIGHT=1 (skip check).")


def _allocatable(m: Model) -> list[int] | None:
    """Whole allocatable CPUs of each schedulable node (m.node only, when pinned); None when nodes can't be read.
    Allocatable already excludes kube/system-reserved; exclusive CPUs cannot span nodes."""
    nodes = k8s.get_json("node", m.node) if m.node else k8s.get_json("nodes")
    if not nodes:
        return None
    return [whole_cpus(text(pick(pick(n.get("status"), "allocatable", {}), "cpu", "0")), floor=True)
            for n in ([nodes] if m.node else nodes.get("items", []))
            if not any(t.get("effect") == "NoSchedule" for t in pick(n.get("spec"), "taints", []))]


def _replica_shortfall(alloc: list[int] | None, per_pod: int, replicas: int) -> str:
    """Each replica needs per_pod CPUs on one node, so count whole replica slots per node."""
    if replicas < 2 or alloc is None or per_pod < 1:  # [] is known: no schedulable node
        return ""
    slots = sum(a // per_pod for a in alloc)
    if replicas > slots:
        return (f"{replicas} replicas x {per_pod} CPUs: the schedulable nodes fit {slots} "
                f"({sum(alloc)} allocatable CPUs; a replica cannot span nodes)")
    return ""


class KubeletStatic(BestEffort):
    """kubelet static CPU manager: Guaranteed-QoS pods (integer cpu, requests == limits) get exclusive cores."""
    name = label = "kubelet-static"
    full_pcpus = False

    def prepare(self, m: Model) -> None:
        cfg = kubelet_config(probe_node(m.node)) or {}
        self.full_pcpus = text(pick(pick(cfg, "cpuManagerPolicyOptions", {}), "full-pcpus-only")) == "true"
        # SMT width is not visible through the API; full-pcpus-only implies the x86 norm of 2.
        tpc = 2 if self.full_pcpus else 1
        tpc_env = os.environ.get("MM_KUBELET_THREADS_PER_CORE", "")
        if re.fullmatch(r"[1-9][0-9]*", tpc_env):
            tpc = int(tpc_env)
        elif tpc_env:
            warn(f"Invalid MM_KUBELET_THREADS_PER_CORE='{tpc_env}' — expected a positive integer; using "
                 "auto-detected default.")
        cpu = whole_cpus(m.cpu)
        if cpu < 1:
            abort(f"kubelet-static needs a numeric CPU quantity >= 1 for '{m.name}', got '{m.cpu}'. Set an integer "
                  "cpu in models.yaml or pass --cpu N.")
        if self.full_pcpus and cpu % tpc:
            cpu += tpc - cpu % tpc  # kubelet rejects partial cores with SMTAlignmentError
        if str(cpu) != m.cpu:
            aligned = f", full-pcpus-only aligned to {tpc}-thread cores" if self.full_pcpus else ""
            info(f"kubelet-static: CPU {m.cpu} → {cpu} (integer, Guaranteed QoS{aligned})")
            m.cpu = str(cpu)
        if not m.memory:
            abort(f"kubelet-static requires an explicit memory value for '{m.name}' (Guaranteed QoS needs requests "
                  "== limits). Set memory in models.yaml or pass --memory.")

    def annotations(self, m: Model, tp: int) -> tuple[dict, str]:
        anns = {f"{ANN}/type": self.name, f"{ANN}/qos": "guaranteed", f"{ANN}/cpus": m.cpu}
        if self.full_pcpus:
            anns[f"{ANN}/full-pcpus-only"] = "true"
        if tp > 1:
            anns[f"{ANN}/tensor-parallel"] = str(tp)
        return anns, m.cpu

    def preflight(self, m: Model, tp: int) -> list[str]:
        node = probe_node(m.node)
        cfg = kubelet_config(node) if node else {}
        if node and pick(cfg, "cpuManagerPolicy") != "static":
            if cfg is None:
                warn(f"Could not read kubelet config from node '{node}' (needs RBAC on nodes/proxy) — cannot confirm "
                     "cpuManagerPolicy=static. Proceeding; if the kubelet is not static the model runs without CPU "
                     "pinning.")
            else:
                warn(f"Node '{node}' has cpuManagerPolicy={text(pick(cfg, 'cpuManagerPolicy', 'none'))} — not "
                     f"'static'. {m.name} will run WITHOUT exclusive CPUs. Re-run the solutions install with "
                     "kubernetes_cpu_policy=kubelet-static to reconfigure kubelet.")
        alloc = _allocatable(m)
        if alloc is None:
            warn("Could not list nodes to verify CPU capacity — skipping the kubelet-static capacity check.")
            return []
        per_pod, largest = int(m.cpu), max(alloc, default=0)  # the pod requests m.cpu; TP splits it
        if largest and per_pod > largest:
            return [(f"No schedulable node has {per_pod} allocatable CPUs (cpu={m.cpu}); the largest offers "
                     f"{largest}. Exclusive CPUs cannot span nodes, so the pod would stay Pending.")]
        shortfall = _replica_shortfall(alloc, per_pod, int(m.replicas))
        return [f"{shortfall}; the replicas that don't fit would stay Pending."] if shortfall else []


    def failure(self, m: Model, tp: int) -> str:
        return (f"kubelet-static pre-flight failed: cannot place {m.name} (cpu={m.cpu} tp={tp} replicas={m.replicas}) "
                "on any single node. Lower --cpu/--tp/--replicas, or add a larger node. Override with MM_PREFLIGHT=warn "
                "(proceed anyway) or MM_SKIP_PREFLIGHT=1 (skip check).")


POLICIES = {p.name: p for p in (NriBalloons, KubeletStatic, BestEffort)}


def whole_cpus(cpu: str, floor: bool = False) -> int:
    """CPU quantity (8, 7.5, 7500m) in whole CPUs, rounded up (or down); 0 when unparseable."""
    if cpu.endswith("m") and is_uint(cpu[:-1]):
        value = int(cpu[:-1]) / 1000
    elif re.fullmatch(r"[0-9]*\.?[0-9]+", cpu):
        value = float(cpu)
    else:
        return 0
    return math.floor(value) if floor else math.ceil(value)


def resolve_policy(requested: str, node: str) -> BestEffort:
    if requested and requested not in POLICIES:
        abort(f"Invalid --cpu-policy '{requested}'. Valid: nri-balloons, kubelet-static, best-effort.")
    env_policy = os.environ.get("MM_CPU_POLICY", "")
    if not requested and env_policy:
        requested = env_policy.lower() if env_policy.lower() in POLICIES else ""
        if not requested:
            warn(f"Invalid MM_CPU_POLICY='{env_policy}' — ignoring, falling back to auto-detection.")
    nri_enabled = os.environ.get("MM_NRI_ENABLED", "")
    if not requested and nri_enabled and not truthy(nri_enabled):
        requested = "best-effort"
        info(f"MM_NRI_ENABLED={nri_enabled} — CPU pinning disabled (best-effort).")
    if not requested:
        node = probe_node(node)
        if node and pick(kubelet_config(node), "cpuManagerPolicy") == "static":
            requested = "kubelet-static"
            info(f"Detected kubelet static CPU manager on '{node}' — using kubelet-static.")
        elif k8s.succeeds("get", "daemonset", "nri-resource-policy-balloons", "-n", "kube-system", "-o", "name"):
            requested = "nri-balloons"
        else:
            requested = "best-effort"
            warn("No CPU-pinning mechanism detected (no NRI balloons DaemonSet, kubelet cpuManagerPolicy is not "
                 "static) — deploying best-effort, without CPU pinning.")
    return POLICIES[requested]()


def resolve_auto_cpu(policy: BestEffort, m: Model, tp: int) -> None:
    """Turn `cpu: auto` (models.yaml) into a number, before the policy normalises m.cpu; --cpu overrides it.

    A deployed model keeps its CPU, so a re-deploy leaves the spec unchanged. Otherwise nri-balloons sizes it
    to the free balloon CPUs of tp NUMA nodes on the smallest schedulable node, and aborts below the category
    default cpu; the other policies have no NUMA balloons to size against and take that default.
    """
    if m.cpu != "auto":
        return
    # An OVMS upgrade from InferenceService keeps the CPU the old one runs with.
    kinds = [m.kind] + (["InferenceService"] if m.kind == "LLMInferenceService" else [])
    live = next((v for k in kinds if (v := k8s.out("get", k, m.name, "-n", m.namespace, "-o",
                                                   f"jsonpath={LIVE_CPU[k]}"))), "")
    if live:
        if isinstance(policy, NriBalloons) and is_uint(live):
            live = str(nri_policy.model_cpu(int(live)))  # the request carries any hideHyperthreads doubling
        info(f"  CPU auto: keeping deployed value {live}")
        m.cpu = live
        return
    default = text(pick(pick(m.cat, "defaults", {}), "cpu", 8))
    if isinstance(policy, NriBalloons):
        cpu = nri_policy.auto_cpu(tp, m.node)
        if cpu < whole_cpus(default):
            abort(f"cpu: auto — only {cpu} free CPUs per NUMA node for {m.name} (tp={tp}, need >= {default}). "
                  "Undeploy a model, add a node, or pass --cpu.")
        m.cpu = str(cpu)
    else:
        m.cpu = default
    info(f"  CPU auto: {m.cpu} (tp={tp}, policy {policy.name})")


def _preflight_skipped() -> bool:
    return truthy(os.environ.get("MM_SKIP_PREFLIGHT", ""))


_MEM_UNITS = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "Pi": 2**50,
              "k": 10**3, "K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12, "P": 10**15}


def mem_bytes(quantity: str) -> int:
    """Memory quantity (16Gi, 512Mi, 2G, 1e9, 1048576) in bytes; 0 when unparseable."""
    q = re.fullmatch(r"([0-9]*\.?[0-9]+)(e[0-9]+|[KMGTP]i|[kKMGTP])?", quantity.strip())
    if not q:
        return 0
    value, unit = float(q.group(1)), q.group(2) or ""
    return int(value * 10 ** int(unit[1:])) if unit.startswith("e") else int(value * _MEM_UNITS.get(unit, 1))


def _cpu_millis(quantity: str) -> int:
    if quantity.endswith("m") and is_uint(quantity[:-1]):
        return int(quantity[:-1])
    return int(float(quantity) * 1000) if re.fullmatch(r"[0-9]*\.?[0-9]+", quantity) else 0


AMX_LABEL = "feature.node.kubernetes.io/cpu-cpuid.AMXBF16"


def _request_errors(m: Model, cpu: str, amx: bool) -> list[str]:
    """The scheduler places a pod only where its CPU and memory requests are free: count, per node the model can
    use, how many replicas fit beside the pods already there."""
    per_cpu, per_mem, replicas = _cpu_millis(cpu), mem_bytes(m.memory), int(m.replicas)
    if replicas < 1 or (per_cpu < 1 and per_mem < 1):
        return []
    nodes = k8s.get_json("node", m.node) if m.node else k8s.get_json("nodes")
    pods = k8s.get_json("pods", "-A", "--field-selector", "status.phase!=Succeeded,status.phase!=Failed")
    if not nodes or pods is None:
        warn("Could not list nodes or pods — skipping the CPU/memory request check.")
        return []
    used: dict[str, list[int]] = {}
    for p in pods.get("items", []):
        node = pick(pick(p, "spec", {}), "nodeName")
        if node:
            total = used.setdefault(node, [0, 0])
            for c in pick(pick(p, "spec", {}), "containers", []):
                req = pick(pick(c, "resources", {}), "requests", {})
                total[0] += _cpu_millis(text(pick(req, "cpu", "0")))
                total[1] += mem_bytes(text(pick(req, "memory", "0")))
    slots, free = 0, []
    for n in [nodes] if m.node else nodes.get("items", []):
        name, labels = n["metadata"]["name"], pick(n["metadata"], "labels", {})
        if any(t.get("effect") == "NoSchedule" for t in pick(n.get("spec"), "taints", [])) or \
                (amx and labels.get(AMX_LABEL) != "true"):
            continue
        alloc, (cpu_used, mem_used) = pick(pick(n, "status", {}), "allocatable", {}), used.get(name, [0, 0])
        free_cpu = _cpu_millis(text(pick(alloc, "cpu", "0"))) - cpu_used
        free_mem = mem_bytes(text(pick(alloc, "memory", "0"))) - mem_used
        fits = [f // need for f, need in ((free_cpu, per_cpu), (free_mem, per_mem)) if need > 0]
        slots += max(0, min(fits))
        free.append(f"{name} {max(0, free_cpu) // 1000} CPU / {max(0, free_mem) // 2**30}Gi")
    if replicas <= slots:
        return []
    where = "; ".join(free) or "no schedulable node" + (" with AMX" if amx else "")
    return [f"{replicas} replica(s) requesting {cpu} CPU and {m.memory} each: the nodes it can use have room for "
            f"{slots} (free: {where}); the rest would stay Pending."]


def preflight_gate(policy: BestEffort, m: Model, tp: int, cpu: str, running: int = 0, amx: bool = False) -> None:
    """Abort, before weights are downloaded, a deploy whose replicas (up to the autoscaling maximum) do not all fit
    now, so none is left Pending. `running` replicas already hold their place (a scaling-only change, or an upgrade
    that frees the old pods first), so only the ones beyond them are checked."""
    if _preflight_skipped():
        return
    total = int(m.max_replicas or m.replicas)
    extra = total - running
    if extra < 1:
        return
    probe = dataclasses.replace(m, replicas=str(extra))
    # A policy that checks allocatable, not free, capacity counts every replica, running ones too.
    placed = probe if policy.counts_running_pods else dataclasses.replace(m, replicas=str(total))
    errors = policy.preflight(placed, tp) + _request_errors(probe, cpu, amx)
    if not errors:
        return
    what = f"{extra} replica(s)" + (f" beyond the {running} running" if running else "") + \
        (f" (autoscaling up to {total})" if m.autoscaled else "")
    warn(f"{policy.label} pre-flight: {m.name} cannot place {what}:")
    for e in errors:
        warn(f"  - {e}")
    if "warn" in (os.environ.get("MM_PREFLIGHT"), os.environ.get("MM_NRI_PREFLIGHT")):
        return warn(f"{policy.name} pre-flight failed but MM_PREFLIGHT=warn — proceeding anyway.")
    abort(policy.failure(placed, tp))
