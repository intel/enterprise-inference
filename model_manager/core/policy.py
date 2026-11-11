"""CPU pinning policies. Precedence: --cpu-policy > MM_CPU_POLICY > MM_NRI_ENABLED=false > cluster auto-detect."""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import re
from typing import Any

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
    name = "best-effort"
    download_nodes: tuple = ()  # nodes the weight-download Job must run on; empty: anywhere

    def prepare(self, m: Model) -> None:
        """Normalise m.cpu/m.memory for the policy; may abort."""

    def annotations(self, m: Model, tp: int) -> tuple[dict, str]:
        """Pod annotations and the CPU quantity to request."""
        return {f"{ANN}/type": self.name}, m.cpu

    def preflight(self, m: Model, tp: int) -> list[str]:
        """Reasons the model cannot be placed (empty when it fits)."""
        return []

    def failure(self, m: Model, tp: int) -> str:
        return ""


class NriBalloons(BestEffort):
    """NRI balloons plugin: NUMA-aware balloons sized per tensor-parallel shard."""
    name = "nri-balloons"
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

    def preflight(self, m: Model, tp: int) -> list[str]:
        if not truthy(os.environ.get("MM_NRI_ENABLED", "true")):
            return []
        if not is_uint(m.cpu):
            return [f"nri-balloons needs a whole number of CPUs, got cpu={m.cpu}"]
        with contextlib.redirect_stderr(io.StringIO()):  # notes are noise here; errors are reported below
            result = nri_policy.preflight(int(m.cpu), tp, node=m.node, replicas=int(m.replicas))
        return [] if result["ok"] else result["errors"]

    def failure(self, m: Model, tp: int) -> str:
        return (f"NRI pre-flight failed: not enough physical cores to place {m.name} (cpu={m.cpu} tp={tp} "
                f"replicas={m.replicas}) without spilling onto reserved hyperthread-sibling cores. Lower "
                "--cpu/--replicas, raise --tp, undeploy a model, or add a node. Override with MM_NRI_PREFLIGHT=warn "
                "(proceed anyway) or MM_NRI_SKIP_PREFLIGHT=1 (skip check).")


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
        cpu = _whole_cpus(m.cpu)
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
        nodes = k8s.get_json("node", m.node) if m.node else k8s.get_json("nodes")
        if not nodes:
            warn("Could not list nodes to verify CPU capacity — skipping the kubelet-static capacity check.")
            return []
        # Allocatable already excludes kube/system-reserved; exclusive CPUs cannot span nodes.
        alloc = [_whole_cpus(text(pick(pick(n.get("status"), "allocatable", {}), "cpu", "0")), floor=True)
                 for n in ([nodes] if m.node else nodes.get("items", []))
                 if not any(t.get("effect") == "NoSchedule" for t in pick(n.get("spec"), "taints", []))]
        per_pod, replicas, largest = int(m.cpu) * tp, int(m.replicas), max(alloc, default=0)
        if replicas > 1 and largest and per_pod * replicas > sum(alloc):
            warn(f"kubelet-static: {replicas} replicas x {per_pod} CPUs = {per_pod * replicas} exceeds the cluster's "
                 f"{sum(alloc)} allocatable CPUs — some replicas will stay Pending.")
        if largest and per_pod > largest:
            return [f"No schedulable node has {per_pod} allocatable CPUs (cpu={m.cpu} x tp={tp}); the largest offers "
                    f"{largest}. Exclusive CPUs cannot span nodes, so the pod would stay Pending."]
        return []

    def failure(self, m: Model, tp: int) -> str:
        return (f"kubelet-static pre-flight failed: cannot place {m.name} (cpu={m.cpu} tp={tp} replicas={m.replicas}) "
                "on any single node. Lower --cpu/--tp/--replicas, or add a larger node. Override with MM_PREFLIGHT=warn "
                "(proceed anyway) or MM_SKIP_PREFLIGHT=1 (skip check).")


class GpuRuntime(BestEffort):
    """device: gpu runtimes carry no CPU pinning: balloons and static-CPU alignment are Xeon placement
    controls, and the accelerator resource limit does the scheduling. The pod is rendered without a CPU
    limit (Burstable), so kubelet's static CPU manager does not pin it either. Never selected by --cpu-policy."""
    name = "n/a (GPU runtime — no CPU pinning)"
    label = "GPU"
    rolls = True  # deploy sets False when only spec.replicas changes: the running pods stay

    def annotations(self, m: Model, tp: int) -> tuple[dict, str]:
        return {}, m.cpu

    def preflight(self, m: Model, tp: int) -> list[str]:
        """The cluster's free GPUs decide, not the declared accelerator: a node in the rack profile that never
        joined, or whose device plugin is not up, advertises none. --node is only a preference, so every node counts."""
        nodes = k8s.get_json("nodes")
        pods = k8s.get_json("pods", "-A", "--field-selector", "status.phase!=Succeeded,status.phase!=Failed")
        if not nodes or not pods:
            warn(f"Could not list nodes and pods to verify free {m.accel_resource} — skipping the GPU capacity check.")
            return []
        resource, tolerated = m.accel_resource, text(pick(m.rt, "accelerator_toleration_key"))
        used, own, own_pods = _gpu_usage(pods.get("items", []), m, resource)
        need, new = int(m.gpus), max(int(m.replicas) - own_pods, 1 if self.rolls else 0)
        if not new:
            return []
        ready = {text(pick(n.get("metadata"), "name")): n for n in nodes.get("items", []) if _schedulable(n, tolerated)}
        alloc = {name: _quantity(pick(pick(n.get("status"), "allocatable", {}), resource)) for name, n in ready.items()}
        if not any(alloc.values()):
            return [f"No Ready, schedulable node advertises allocatable {resource}: no GPU node has joined the "
                    "cluster, or its device plugin is not ready yet."]
        free = {name: count - used.get(name, 0) - own.get(name, 0) for name, count in alloc.items() if count}
        pvc, terms, unbound = _model_store(m)
        if terms:
            free = {name: f for name, f in free.items() if _node_matches(ready[name], terms)}
        per_node = [max(f, 0) // need for f in free.values()]
        # A store that binds on first use lands on one node, and every replica must mount it there.
        fit = max(per_node, default=0) if unbound else sum(per_node)
        if not fit and self.rolls and own_pods and any(alloc[name] - used.get(name, 0) >= need for name in free):
            return [f"{m.name}'s own running pods hold the {resource} it needs: if this change restarts the pod (any "
                    "change but --replicas may), the rollout starts the new one next to the running one. Undeploy it "
                    f"first (./model-manager undeploy {m.name}), or use MM_PREFLIGHT=warn to apply anyway (e.g. a "
                    "label-only change)."]
        if not fit and terms:
            return [f"The model store {pvc} is node-local storage on nodes without {need} free {resource}, so the "
                    "GPU pod could not mount it. GPU models need shared (ReadWriteMany) storage, or the model store "
                    "on a GPU node."]
        if not fit:
            return [f"No node has {need} free {resource}; the most on one node is {max(max(free.values()), 0)}. A "
                    "pod's GPUs cannot span nodes: lower --gpus, or free GPUs on a node."]
        if fit < new:
            warn(f"GPU: only {fit} of {new} new replicas fit on the free {resource} — the rest will stay Pending.")
        if unbound:
            self.download_nodes = tuple(name for name, f in free.items() if f >= need)
            warn(f"The model store {pvc} binds on first use, so it is created on one of {', '.join(self.download_nodes)}"
                 " where the download runs. On node-local storage, every model using it, CPU ones included, must then "
                 "run on that node.")
        return []

    def failure(self, m: Model, tp: int) -> str:
        return (f"GPU pre-flight failed: cannot place {m.name} ({m.gpus} x {m.accel_resource}) on any node. Override "
                "with MM_PREFLIGHT=warn (proceed anyway) or MM_SKIP_PREFLIGHT=1 (skip check).")


POLICIES = {p.name: p for p in (NriBalloons, KubeletStatic, BestEffort)}


def _gpu_usage(pods: list, m: Model, resource: str) -> tuple[dict, dict, int]:
    """GPUs held per node by other pods and by m's own pods, and how many of m's scheduled pods hold GPUs."""
    used: dict = {}
    own: dict = {}
    own_pods = 0
    for pod in pods:
        meta, node = pod.get("metadata") or {}, text(pick(pod.get("spec"), "nodeName"))
        mine = (pick(meta, "namespace"), pick(pick(meta, "labels", {}), "app.kubernetes.io/name")) == (m.namespace, m.name)
        gpus = _pod_request(pod, resource)
        held = own if mine else used
        held[node] = held.get(node, 0) + gpus
        own_pods += bool(mine and gpus and node)
    return used, own, own_pods


def _pod_request(pod: dict, resource: str) -> int:
    """A pod's effective request, as the scheduler counts it: its containers plus restartable init (sidecar)
    containers, or the peak while the init containers run, whichever is larger."""
    def request(c: dict) -> int:
        return _quantity(pick(pick(c.get("resources"), "requests", {}), resource))
    spec, sidecars, peak = pod.get("spec") or {}, 0, 0
    for c in pick(spec, "initContainers", []):
        if c.get("restartPolicy") == "Always":
            sidecars += request(c)
            peak = max(peak, sidecars)
        else:
            peak = max(peak, sidecars + request(c))
    return max(sum(request(c) for c in pick(spec, "containers", [])) + sidecars, peak)


def _model_store(m: Model) -> tuple[str, list, bool]:
    """The model-store PVC, its PV's required node terms (node-local storage), and whether it is unbound and
    binds on first use, so the download Job's node decides where it lands."""
    pvc = m.uri.split("://", 1)[1].split("/", 1)[0] if m.uri.startswith("pvc://") else ""
    claim = k8s.get_json("pvc", pvc, "-n", m.namespace) if pvc else None
    if not claim:
        return pvc, [], False
    volume = text(pick(claim.get("spec"), "volumeName"))
    if volume:
        affinity = pick(pick(k8s.get_json("pv", volume), "spec", {}), "nodeAffinity", {})
        return pvc, pick(pick(affinity, "required", {}), "nodeSelectorTerms", []), False
    mode = k8s.out("get", "storageclass", text(pick(claim.get("spec"), "storageClassName")), "-o",
                   "jsonpath={.volumeBindingMode}")
    return pvc, [], mode == "WaitForFirstConsumer"


def _node_matches(node: dict, terms: list) -> bool:
    """Whether a node satisfies required nodeSelectorTerms: any term, with all of its expressions."""
    labels, name = pick(node.get("metadata"), "labels", {}), text(pick(node.get("metadata"), "name"))

    def holds(e: dict, value: Any) -> bool:
        values = e.get("values") or []
        return {"In": value in values, "NotIn": value not in values, "Exists": value is not None,
                "DoesNotExist": value is None}.get(e.get("operator"), False)
    return any((pick(t, "matchExpressions", []) or pick(t, "matchFields", []))  # an empty term matches no node
               and all(holds(e, labels.get(e.get("key"))) for e in pick(t, "matchExpressions", []))
               and all(holds(f, name if f.get("key") == "metadata.name" else None) for f in pick(t, "matchFields", []))
               for t in terms)


def _quantity(value: Any) -> int:
    """A whole-number extended-resource quantity (GPUs); 0 when absent or unparseable."""
    return int(text(value)) if is_uint(text(value)) else 0


def _schedulable(node: dict, tolerated: str) -> bool:
    """Ready, not cordoned, and no NoSchedule/NoExecute taint other than the accelerator's own, which GPU pods
    tolerate."""
    ready = any(c.get("type") == "Ready" and c.get("status") == "True"
                for c in pick(node.get("status"), "conditions", []))
    blocking = [t for t in pick(node.get("spec"), "taints", []) if t.get("effect") in ("NoSchedule", "NoExecute")
                and not (t.get("key") == tolerated and t.get("effect") == "NoSchedule")]
    return ready and not pick(node.get("spec"), "unschedulable", False) and not blocking


def _whole_cpus(cpu: str, floor: bool = False) -> int:
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
    path = LIVE_CPU.get(m.kind)
    live = k8s.out("get", m.kind, m.name, "-n", m.namespace, "-o", f"jsonpath={path}") if path else ""
    if live:
        if isinstance(policy, NriBalloons) and is_uint(live):
            live = str(nri_policy.model_cpu(int(live)))  # the request carries any hideHyperthreads doubling
        info(f"  CPU auto: keeping deployed value {live}")
        m.cpu = live
        return
    default = text(pick(pick(m.cat, "defaults", {}), "cpu", 8))
    if isinstance(policy, NriBalloons):
        cpu = nri_policy.auto_cpu(tp, m.node)
        if cpu < _whole_cpus(default):
            abort(f"cpu: auto — only {cpu} free CPUs per NUMA node for {m.name} (tp={tp}, need >= {default}). "
                  "Undeploy a model, add a node, or pass --cpu.")
        m.cpu = str(cpu)
    else:
        m.cpu = default
    info(f"  CPU auto: {m.cpu} (tp={tp}, policy {policy.name})")


def preflight_gate(policy: BestEffort, m: Model, tp: int) -> None:
    """Abort a deploy that cannot be placed, before weights are downloaded."""
    if truthy(os.environ.get("MM_SKIP_PREFLIGHT", "")):
        return
    errors = policy.preflight(m, tp)
    if not errors:
        return
    warn(f"{policy.label} pre-flight issues detected:")
    for e in errors:
        warn(f"  - {e}")
    if "warn" in (os.environ.get("MM_PREFLIGHT"), os.environ.get("MM_NRI_PREFLIGHT")):
        return warn(f"{policy.name} pre-flight failed but MM_PREFLIGHT=warn — proceeding anyway.")
    abort(policy.failure(m, tp))
