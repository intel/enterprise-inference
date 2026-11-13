# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NRI Balloons CPU policy — standalone helper for model-manager v2.

Works with the policy the nri_cpu_balloons role installs:
  - model-balloon-tp<N> balloon types for any engine; hyperthread siblings reserved, so a CPU request is
    physical cores
  - pre-flight placement on each node's own NUMA layout (nri.intel.com/numa-cpus), the NUMA sets its
    BalloonsPolicy gives each balloon type, and the free CPUs NRI reports (NodeResourceTopology)

Used by core/policy.py for the nri-balloons CPU policy. No pip dependencies —
uses kubectl subprocess calls for all K8s access; diagnostics go to stderr.
"""

from __future__ import annotations

import fnmatch
import itertools
import json
import math
import os
import re
# Only used for fixed, non-shell kubectl calls (see _kubectl below).
import subprocess  # nosec B404
import sys
from typing import Any

# ── Constants ────────────────────────────────────────────────────────────────

_NRI_BALLOONS_DS = "nri-resource-policy-balloons"
_NRI_BALLOONS_NS = "kube-system"
_NRI_BALLOON_ANNOTATION = "balloon.balloons.resource-policy.nri.io/container.main"
_NRI_BALLOON_POD_ANNOTATION = "balloon.balloons.resource-policy.nri.io/pod"
_NRI_MAX_TP_LABEL = "nri.intel.com/max-tp"
_NRI_ANNOTATION_PREFIX = "balloon.balloons.resource-policy.nri.io/"
_DEFAULT_BALLOON = "model-balloon"  # the role's model-balloon-tp<N> types, any engine
_BALLOONSPOLICY_CRD = "balloonspolicies.config.nri"
_NRT_CRD = "noderesourcetopologies.topology.node.k8s.io"
_NUMA_CPUS_ANNOTATION = "nri.intel.com/numa-cpus"  # {"0": "0-85,172-257", ...}, from the nri_cpu_balloons role
_POD_LABEL_PREFIX = "pod/labels/"

# ── Logging ──────────────────────────────────────────────────────────────────

def _info(msg: str) -> None:
    print(f"[INFO ]  {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    print(f"[WARN ]  {msg}", file=sys.stderr)


# ── kubectl wrapper ──────────────────────────────────────────────────────────

def _kubectl(*args: str) -> str:
    """Run kubectl and return stdout; '' on any error."""
    try:
        # "kubectl" is a fixed argv[0] resolved from the system executable
        # search path (no shell), and args originate from this module's own
        # trusted constants / CLI parameters, not from untrusted external input.
        result = subprocess.run(  # nosec B603 B607
            ["kubectl", *args],
            capture_output=True, text=True, timeout=30,
            shell=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _kubectl_json(*args: str) -> Any:
    """Run kubectl with -o json and parse the result."""
    raw = _kubectl(*args, "-o", "json")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ── NRI verification & policy detection ──────────────────────────────────────

def _nri_balloons_installed() -> bool:
    """Check if the NRI balloons DaemonSet exists in kube-system."""
    raw = _kubectl(
        "get", "daemonset", _NRI_BALLOONS_DS,
        "-n", _NRI_BALLOONS_NS,
        "-o", "name",
    )
    return bool(raw)


def report_daemonset() -> None:
    """Log whether the NRI balloons DaemonSet is deployed; pods are annotated either way."""
    if _nri_balloons_installed():
        _info("Detected NRI balloons DaemonSet — using nri-balloons policy")
    else:
        _warn(
            "NRI balloons DaemonSet not found in kube-system. "
            "Pods will be annotated for NRI but will not receive CPU pinning "
            "until the nri-resource-policy-balloons DaemonSet is deployed. "
            "Run the nri_cpu_balloons Ansible role to install it."
        )


# ── Balloon name resolution ─────────────────────────────────────────────────

def balloon_name(tp: int = 1, balloon_override: str = "") -> str:
    """The balloon type a tp-N model binds to: MM_NRI_BALLOON_NAME, an explicit override, else the
    nri_cpu_balloons role's model-balloon-tp<N> (N rounded up to 1, 2, 4 or 8)."""
    env_name = os.environ.get("MM_NRI_BALLOON_NAME", "").strip()
    if env_name:
        return env_name
    if balloon_override:
        return balloon_override
    return f"{_DEFAULT_BALLOON}-tp{next((size for size in (1, 2, 4, 8) if tp <= size), 8)}"


# ── Annotations ──────────────────────────────────────────────────────────────

def build_annotations(
    tp: int,
    kind: str,
    balloon_override: str = "",
) -> dict[str, str]:
    """Build NRI balloon annotations for the manifest.

    Returns a dict of annotation key→value pairs. The balloon binding
    annotation is included so the NRI plugin picks it up.
    """
    bname = balloon_name(tp, balloon_override)
    ann_key = _NRI_BALLOON_POD_ANNOTATION if kind == "InferenceService" else _NRI_BALLOON_ANNOTATION
    annotations: dict[str, str] = {
        "cpu-policy.model-manager.io/type": "nri-balloons",
        "cpu-policy.model-manager.io/balloon": bname,
        ann_key: bname,
    }
    _info(f"Binding pods to NRI balloon '{bname}'")
    return annotations


# ── NUMA topology helpers ───────────────────────────────────────────────────

def _node_label(node: str, label: str) -> str:
    escaped = label.replace(".", "\\.")
    return _kubectl(
        "get", "node", node,
        "-o", f"jsonpath={{.metadata.labels.{escaped}}}",
    )


def _parse_cpuset(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            try:
                out.update(range(int(lo), int(hi) + 1))
            except ValueError:
                continue
        else:
            try:
                out.add(int(part))
            except ValueError:
                continue
    return out


def _node_max_tp(node: str) -> int:
    raw = _node_label(node, _NRI_MAX_TP_LABEL)
    try:
        return int(raw) if raw else 0
    except (ValueError, TypeError):
        return 0


def label_balloon_types(policies: Any) -> list[tuple[str, list[dict]]]:
    """Balloon types that select pods by label (matchExpressions on pod/labels/...), from a
    BalloonsPolicy list. OpenShift AI's v1alpha1 LLMInferenceService cannot annotate its pods, so
    they land in their balloon this way. Node-specific policies repeat the default's types."""
    types, seen = [], set()
    for policy in (policies or {}).get("items", []):
        for bt in (policy.get("spec") or {}).get("balloonTypes") or []:
            exprs = bt.get("matchExpressions") or []
            if bt["name"] in seen or not exprs or not all(e.get("key", "").startswith(_POD_LABEL_PREFIX)
                                                          for e in exprs):
                continue
            seen.add(bt["name"])
            types.append((bt["name"], exprs))
    return types


def label_matches(labels: dict, exprs: list[dict]) -> bool:
    """Whether pod labels satisfy a balloon type's pod/labels/ matchExpressions, with the NRI
    plugins' semantics (resmgr Expression.Evaluate): "*" in Equals/In matches any value, and the
    Matches* operators take glob patterns."""
    for e in exprs:
        op, values = e.get("operator"), e.get("values") or [""]
        if op == "AlwaysTrue":
            continue
        value = labels.get(e["key"][len(_POD_LABEL_PREFIX):])
        has = value is not None
        globbed = has and any(fnmatch.fnmatchcase(value, v) for v in values)
        ok = {"Equals": has and values[0] in (value, "*"), "NotEqual": not has or value != values[0],
              "In": has and any(v in (value, "*") for v in values),
              "NotIn": not (has and any(v in (value, "*") for v in values)),
              "Matches": has and fnmatch.fnmatchcase(value, values[0]),
              "MatchesNot": not (has and fnmatch.fnmatchcase(value, values[0])),
              "MatchesAny": globbed, "MatchesNone": not globbed,
              "Exists": has, "NotExist": not has}.get(op, False)
        if not ok:
            return False
    return True


def _balloon_pods_cpu_on_node(node: str) -> int:
    """Sum CPU requests of the pods on a node that a balloon takes: annotated for one, or
    matching a balloon type's pod labels (OpenShift AI's v1alpha1 models)."""
    pods = _kubectl_json("get", "pods", "-A",
                         "--field-selector", f"spec.nodeName={node}")
    if not pods or "items" not in pods:
        return 0
    label_types = label_balloon_types(_kubectl_json("get", _BALLOONSPOLICY_CRD, "-A"))
    total_millis = 0
    for pod in pods["items"]:
        phase = pod.get("status", {}).get("phase", "")
        if phase not in ("Running", "Pending"):
            continue
        anns = pod.get("metadata", {}).get("annotations") or {}
        labels = pod.get("metadata", {}).get("labels") or {}
        if not any(k.startswith(_NRI_ANNOTATION_PREFIX) for k in anns) and not any(
                label_matches(labels, exprs) for _, exprs in label_types):
            continue
        for c in pod.get("spec", {}).get("containers") or []:
            req_cpu = ((c.get("resources") or {}).get("requests") or {}).get("cpu", "0")
            millis = _cpu_str_to_millis(str(req_cpu))
            total_millis += millis
    return total_millis // 1000


def _cpu_str_to_millis(s: str) -> int:
    s = s.strip()
    if not s:
        return 0
    if s.endswith("m"):
        try:
            return int(s[:-1])
        except ValueError:
            return 0
    try:
        return int(float(s) * 1000)
    except ValueError:
        return 0


# ── Cluster-wide TP capability ───────────────────────────────────────────────

def _cluster_max_tp() -> tuple[int, int]:
    """Scan nodes for nri.intel.com/max-tp.

    Returns (best_max_tp, labeled_node_count).
    """
    nodes = _kubectl_json("get", "nodes")
    if not nodes or "items" not in nodes:
        return (0, 0)
    best = 0
    labeled = 0
    for node in nodes["items"]:
        labels = node.get("metadata", {}).get("labels") or {}
        raw = labels.get(_NRI_MAX_TP_LABEL)
        if raw is None:
            continue
        labeled += 1
        spec = node.get("spec", {})
        if spec.get("unschedulable"):
            continue
        ready = any(
            c.get("type") == "Ready" and c.get("status") == "True"
            for c in (node.get("status", {}).get("conditions") or [])
        )
        if not ready:
            continue
        try:
            val = int(str(raw).strip())
        except (ValueError, TypeError):
            continue
        if val > best:
            best = val
    return (best, labeled)


def _schedulable_nri_nodes(min_tp: int) -> list[str]:
    """Ready, schedulable nodes with max-tp >= min_tp."""
    nodes = _kubectl_json("get", "nodes")
    if not nodes or "items" not in nodes:
        return []
    out: list[str] = []
    for node in nodes["items"]:
        labels = node.get("metadata", {}).get("labels") or {}
        raw = labels.get(_NRI_MAX_TP_LABEL)
        if raw is None:
            continue
        spec = node.get("spec", {})
        if spec.get("unschedulable"):
            continue
        ready = any(
            c.get("type") == "Ready" and c.get("status") == "True"
            for c in (node.get("status", {}).get("conditions") or [])
        )
        if not ready:
            continue
        try:
            cap = int(str(raw).strip())
        except (ValueError, TypeError):
            continue
        if cap >= min_tp:
            out.append(node["metadata"]["name"])
    return out


# ── Per-NUMA capacity analysis ───────────────────────────────────────────────

def _reserved_cpuset(node: str) -> set[int]:
    """The reserved cpuset of the node's BalloonsPolicy (empty when given as a count)."""
    for policy in (f"node.{node}", "default"):
        raw = _kubectl("get", _BALLOONSPOLICY_CRD, policy, "-n", _NRI_BALLOONS_NS,
                       "-o", "jsonpath={.spec.reservedResources.cpu}")
        if raw:
            return _parse_cpuset(raw[7:]) if raw.startswith("cpuset:") else set()
    return set()


def _numa_cpusets(node: str) -> dict[int, set[int]]:
    """Each NUMA node's logical CPUs, as the nri_cpu_balloons role published them from the node's sysfs
    ({} when not published: no CPU layout is assumed)."""
    raw = _kubectl("get", "node", node, "-o", "jsonpath={.metadata.annotations.nri\\.intel\\.com/numa-cpus}")
    try:
        numa = json.loads(raw) if raw else {}
        return {int(k): _parse_cpuset(v) for k, v in numa.items()} if isinstance(numa, dict) else {}
    except (ValueError, TypeError, AttributeError):
        _warn(f"nri-preflight: unreadable {_NUMA_CPUS_ANNOTATION} on '{node}'")
        return {}


def _per_numa_free(node: str, in_use: int) -> dict[int, int] | None:
    """Free balloon CPUs per NUMA node: its CPUs minus the reserved set and every balloon's cpuset, as NRI reports
    them in NodeResourceTopology. Pods whose balloon is not reported yet (`in_use` covers them) are taken from the
    freest NUMA node, where the allocator would put them. None when the topology is unknown."""
    numa = _numa_cpusets(node)
    nrt = _kubectl_json("get", _NRT_CRD, node) if numa else None
    if not nrt:
        return None
    taken: set[int] = set()
    for zone in nrt.get("zones") or []:
        if zone.get("type") == "balloon" and not zone.get("name", "").startswith(("reserved[", "default[")):
            attrs = {a.get("name"): a.get("value", "") for a in zone.get("attributes") or []}
            taken |= _parse_cpuset(attrs.get("cpuset", ""))
    reserved = _reserved_cpuset(node)
    free = {n: len(cpus - reserved - taken) for n, cpus in numa.items()}
    unreported = max(0, in_use - len(taken))  # taken counts a spilled balloon's reserved CPUs too, as its request does
    while unreported > 0 and max(free.values()) > 0:
        n = max(free, key=free.get)
        step = min(unreported, free[n])
        free[n] -= step
        unreported -= step
    return free


def _tp_numa_sets(node: str, balloon: str) -> list[tuple[int, ...]] | None:
    """The NUMA node sets a `balloon` can take, from the node's BalloonsPolicy: a type with components takes one of
    them (componentCreation: balance-balloons) or all of them (otherwise); a leaf type is the NUMA node it is
    close to (preferCloseToDevices: .../node/nodeN). None when the policy does not resolve that far."""
    policy = None
    for name in (f"node.{node}", "default"):
        policy = _kubectl_json("get", _BALLOONSPOLICY_CRD, name, "-n", _NRI_BALLOONS_NS)
        if policy:
            break
    types = {t.get("name"): t for t in ((policy or {}).get("spec") or {}).get("balloonTypes") or []}

    def sets(name: str, depth: int = 0) -> list[tuple[int, ...]] | None:
        t = types.get(name)
        if t is None or depth > 16:
            return None
        parts = [c.get("balloonType", "") for c in t.get("components") or []]
        if not parts:
            numa = {int(m.group(1)) for d in t.get("preferCloseToDevices") or []
                    if (m := re.search(r"/node/node(\d+)/?$", d))}
            return [tuple(numa)] if len(numa) == 1 else None
        children = [sets(c, depth + 1) for c in parts]
        if any(c is None for c in children):
            return None
        if t.get("componentCreation") == "balance-balloons":
            return [s for c in children for s in c]
        return [tuple(sorted({n for s in combo for n in s})) for combo in itertools.product(*children)]

    found = sets(balloon)
    return sorted(set(found)) if found else None


def _largest_replica(node: str, tp: int) -> int | None:
    """The most CPUs one tp-N replica can get on the empty node: its best NUMA set, split evenly, without the
    reserved CPUs. None when the layout or the balloon types are not known."""
    numa, numa_sets = _numa_cpusets(node), _tp_numa_sets(node, balloon_name(tp))
    if not numa or not numa_sets:
        return None
    usable = {n: len(cpus - _reserved_cpuset(node)) for n, cpus in numa.items()}
    return max(min(usable.get(n, 0) for n in s) * len(s) for s in numa_sets)


def _place(free: dict[int, int], numa_sets: list[tuple[int, ...]], cpu: int, want: int) -> int:
    """How many of `want` replicas of `cpu` CPUs fit, each split evenly over one of `numa_sets`; like the
    balloon allocator, each goes to the set with the most free CPUs left."""
    placed = 0
    while placed < want:
        fits = [s for s in numa_sets if all(free.get(n, 0) >= math.ceil(cpu / len(s)) for n in s)]
        if not fits:
            break
        best = max(fits, key=lambda s: min(free[n] for n in s))
        for n in best:
            free[n] -= math.ceil(cpu / len(best))
        placed += 1
    return placed


_REINSTALL = "re-run the nri_cpu_balloons role"


def _node_capacity_report(node: str, cpu: int, tp: int, replicas: int = 1) -> tuple[bool, int, str]:
    """(all replicas fit, how many fit, detail) on one node: replicas placed on the NUMA sets of the node's balloon
    type, against the free CPUs of each NUMA node."""
    replicas, balloon = max(1, replicas), balloon_name(tp)
    free = _per_numa_free(node, _balloon_pods_cpu_on_node(node))
    numa_sets = _tp_numa_sets(node, balloon) if free is not None else None
    if free is None or not numa_sets:
        missing = (f"published NUMA layout ({_NUMA_CPUS_ANNOTATION}, {_NRT_CRD})" if free is None
                   else f"balloon type '{balloon}' in its BalloonsPolicy")
        return False, 0, f"{node}: no {missing}: {_REINSTALL}"
    slots = _place(dict(free), numa_sets, cpu, replicas)
    detail = (f"{node}: need {replicas} replica(s) of {cpu} (tp={tp}, NUMA sets {numa_sets}); node fits {slots} "
              f"(per-NUMA free={[free[n] for n in sorted(free)]})")
    return slots >= replicas, slots, detail


def auto_cpu(tp: int, node: str = "") -> int:
    """CPU for a `cpu: auto` model: the most one replica can still get on the node with the least room, each
    node's best NUMA set split evenly (models already running keep theirs). 0 = nothing free or nothing known."""
    sizes: list[int] = []
    for n in [node] if node else _schedulable_nri_nodes(tp):
        free = _per_numa_free(n, _balloon_pods_cpu_on_node(n))
        numa_sets = _tp_numa_sets(n, balloon_name(tp)) if free is not None else None
        if free is None or not numa_sets:
            _warn(f"nri-preflight: '{n}' has no published NUMA layout or balloon type for tp={tp}: {_REINSTALL}")
            continue
        sizes.append(max(min(free.get(x, 0) for x in s) * len(s) for s in numa_sets))
    return min(sizes) if sizes else 0


# ── Pre-flight validation ───────────────────────────────────────────────────

def preflight(cpu: int, tp: int, node: str = "", replicas: int = 1) -> dict:
    """Run comprehensive pre-deploy NRI validation.

    `replicas` is the model's data-parallel replica count. Each replica is a
    separate pod consuming its own balloon, so the capacity check validates
    cpu * replicas of aggregate demand against the balloon pool.

    Returns: {"ok": bool, "errors": [...], "notes": [...],
              "effective_cpu": int, "balloon": str}
    """
    replicas = max(1, int(replicas or 1))
    mode = os.environ.get("MM_NRI_PREFLIGHT", "").strip().lower()
    skip = os.environ.get("MM_NRI_SKIP_PREFLIGHT", "").strip()
    if skip in ("1", "true", "yes") or mode in ("off", "0", "false", "no"):
        return {"ok": True, "errors": [], "notes": ["preflight disabled"], "skipped": True}

    errors: list[str] = []
    notes: list[str] = []

    # Verify NRI DaemonSet is present
    if not _nri_balloons_installed():
        errors.append(
            "NRI balloons DaemonSet not found in kube-system. "
            "Run the nri_cpu_balloons Ansible role to deploy it."
        )

    eff_cpu = cpu  # siblings are reserved: a request is physical cores
    bname = balloon_name(tp)

    # ── TP capability check ──
    if node:
        node_cap = _node_max_tp(node)
        if node_cap > 0 and tp > node_cap:
            errors.append(
                f"target node '{node}' advertises nri.intel.com/max-tp={node_cap} "
                f"but model requires tp={tp}. "
                f"Lower --tensor-parallel-size to <= {node_cap} or use a node with more NUMA nodes."
            )
        elif node_cap > 0:
            notes.append(f"node '{node}' max-tp={node_cap} (ok for tp={tp})")
    else:
        best, labeled = _cluster_max_tp()
        if labeled == 0:
            notes.append(
                "no node carries nri.intel.com/max-tp; skipping cluster TP check "
                "(re-run the nri_cpu_balloons role to populate node labels)"
            )
        elif best < tp:
            errors.append(
                f"no schedulable node satisfies tp={tp}: best max-tp={best} "
                f"across {labeled} labeled node(s). "
                f"Lower --tensor-parallel-size or add a node with >= {tp} NUMA nodes."
            )
        else:
            notes.append(f"cluster max-tp={best} across {labeled} labeled node(s) (ok for tp={tp})")

    # ── NUMA topology: one replica must fit an empty node ──
    nodes = [node] if node else _schedulable_nri_nodes(tp)
    largest = {n: _largest_replica(n, tp) for n in nodes}
    unknown = [n for n, size in largest.items() if size is None]
    if unknown:
        errors.append(f"no published NUMA layout or balloon type '{bname}' on {', '.join(unknown)}: {_REINSTALL}")
    known = {n: size for n, size in largest.items() if size is not None}
    if known and eff_cpu > max(known.values()):
        errors.append(f"no node can hold one replica of {eff_cpu} CPUs (tp={tp}, balloon '{bname}'): the largest "
                      f"fits {max(known.values())} ({', '.join(f'{n}: {v}' for n, v in known.items())})")

    # ── Capacity: replicas placed on each node's free CPUs per NUMA node ──
    if eff_cpu > 0:
        if node:
            # Pinned to one node: that node alone must hold every replica.
            fits, _slots, detail = _node_capacity_report(node, eff_cpu, tp, replicas)
            (notes if fits else errors).append(f"capacity: {detail}")
        else:
            candidates = _schedulable_nri_nodes(tp)
            if not candidates:
                notes.append(
                    "no schedulable node carries nri.intel.com/max-tp; skipping "
                    "per-node capacity fit (re-run nri_cpu_balloons role)"
                )
            else:
                # No node pin: the k8s scheduler may spread replicas across any
                # subset of eligible nodes. A node contributes as many replica
                # "slots" as its own balloon pool allows (0 if a single replica
                # cannot even fit its NUMA shards there). The deploy fits iff the
                # cluster-wide slot total covers every replica. This is correct
                # for single-node, multi-worker, and multi-master+worker layouts,
                # and heterogeneity-safe because each node is measured from its
                # own labels / reserved seed.
                reports = [
                    _node_capacity_report(n, eff_cpu, tp, replicas)
                    for n in candidates
                ]
                total_slots = sum(s for _f, s, _d in reports)
                if total_slots >= replicas:
                    usable_nodes = sum(1 for _f, s, _d in reports if s > 0)
                    notes.append(
                        f"capacity: {replicas} replica(s) of eff_cpu={eff_cpu} tp={tp} "
                        f"fit across {usable_nodes}/{len(candidates)} schedulable node(s) "
                        f"(cluster slots={total_slots})"
                    )
                else:
                    details = "; ".join(d for _f, _s, d in reports)
                    errors.append(
                        f"cluster cannot host {replicas} replica(s) of eff_cpu={eff_cpu} "
                        f"tp={tp} (balloon='{bname}'): only {total_slots} balloon slot(s) "
                        f"free across {len(candidates)} node(s). {details}. "
                        f"Lower cpu/--replicas, raise --tp, undeploy a model, or add a node."
                    )

    for n in notes:
        _info(f"nri-preflight: {n}")

    ok = len(errors) == 0
    if not ok:
        for e in errors:
            _warn(f"nri-preflight: {e}")
        if mode == "warn":
            _warn("nri-preflight: continuing because MM_NRI_PREFLIGHT=warn")
            ok = True

    return {
        "ok": ok,
        "errors": errors,
        "notes": notes,
        "effective_cpu": eff_cpu,
        "balloon": bname,
    }
