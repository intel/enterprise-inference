#!/usr/bin/env python3
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NRI Balloons CPU policy — standalone helper for model-manager v2.

Implements the proven release-1 NRI balloon policy approach:
  - Verify NRI balloons DaemonSet is deployed
  - Balloon name resolution: vllm-balloon / vllm-balloon-tp<N> (advanced mode)
  - Effective CPU request with hyperthread-hiding doubling
  - Comprehensive pre-flight validation (per-NUMA shard placement)

This module assumes NRI balloons is the only supported CPU policy.
No pip dependencies — uses kubectl subprocess calls for all K8s access.

CLI usage (called from nri.sh / model-manager):
  python3 nri_policy.py detect   [--node NODE]
  python3 nri_policy.py resolve  --cpu CPU --tp TP [--node NODE] [--kind KIND] \
                                 [--balloon NAME]
  python3 nri_policy.py preflight --cpu CPU --tp TP [--node NODE]

All output is JSON on stdout; diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import json
import math
import os
# Only used for fixed, non-shell kubectl calls (see _kubectl below).
import subprocess  # nosec B404
import sys
from typing import Any

# ── Constants ────────────────────────────────────────────────────────────────

_NRI_BALLOONS_DS = "nri-resource-policy-balloons"
_NRI_BALLOONS_NS = "kube-system"
_NRI_POLICY_CONFIGMAP = "nri-resource-policy-balloons-config"
_NRI_BALLOON_ANNOTATION = "balloon.balloons.resource-policy.nri.io/container.main"
_NRI_BALLOON_POD_ANNOTATION = "balloon.balloons.resource-policy.nri.io/pod"
_NRI_MAX_TP_LABEL = "nri.intel.com/max-tp"
_NRI_ANNOTATION_PREFIX = "balloon.balloons.resource-policy.nri.io/"
_DEFAULT_BALLOON = "vllm-balloon"
_BALLOONSPOLICY_CRD = "balloonspolicies.config.nri"

# ── Logging ──────────────────────────────────────────────────────────────────

def _info(msg: str) -> None:
    print(f"[INFO ]  {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    print(f"[WARN ]  {msg}", file=sys.stderr)


# ── kubectl wrapper ──────────────────────────────────────────────────────────

def _kubectl(*args: str, ignore_errors: bool = False) -> str:
    """Run kubectl and return stdout. Returns '' on error if ignore_errors."""
    try:
        # "kubectl" is a fixed argv[0] resolved from the system executable
        # search path (no shell), and args originate from this module's own
        # trusted constants / CLI parameters, not from untrusted external input.
        result = subprocess.run(  # nosec B603 B607
            ["kubectl", *args],
            capture_output=True, text=True, timeout=30,
            shell=False,
        )
        if result.returncode != 0:
            if ignore_errors:
                return ""
            return ""
        return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _kubectl_json(*args: str) -> Any:
    """Run kubectl with -o json and parse the result."""
    raw = _kubectl(*args, "-o", "json", ignore_errors=True)
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
        ignore_errors=True,
    )
    return bool(raw)


def detect_policy(node: str = "") -> str:
    """Detect the active CPU policy.

    Currently only NRI balloons is supported. This function verifies that
    the NRI DaemonSet is present and returns the policy string.

    Returns: "nri-balloons" (always — this is the only supported policy).
    Emits a warning if the NRI DaemonSet is not found.
    """
    if _nri_balloons_installed():
        _info("Detected NRI balloons DaemonSet — using nri-balloons policy")
    else:
        _warn(
            "NRI balloons DaemonSet not found in kube-system. "
            "Pods will be annotated for NRI but will not receive CPU pinning "
            "until the nri-resource-policy-balloons DaemonSet is deployed. "
            "Run the nri_cpu_balloons Ansible role to install it."
        )
    return "nri-balloons"


def verify_nri_installed() -> bool:
    """Verify that the NRI balloons plugin is deployed.

    Returns True if installed. Emits a warning if not found — deployment
    will proceed (annotations are still applied) but pods may not get
    CPU pinning until the NRI DaemonSet is available.
    """
    installed = _nri_balloons_installed()
    if not installed:
        _warn(
            "NRI balloons DaemonSet not found in kube-system. "
            "Pods will be annotated for NRI but will not receive CPU pinning "
            "until the nri-resource-policy-balloons DaemonSet is deployed. "
            "Run the nri_cpu_balloons Ansible role to install it."
        )
    return installed


# ── Advanced mode + HT detection ────────────────────────────────────────────

def _read_nri_configmap_data() -> str:
    """Read the values.yaml from the NRI balloons ConfigMap."""
    raw = _kubectl(
        "get", "configmap", _NRI_POLICY_CONFIGMAP,
        "-n", _NRI_BALLOONS_NS,
        "-o", "jsonpath={.data.values\\.yaml}",
        ignore_errors=True,
    )
    return raw or ""


_cm_data_cache: dict[str, str] = {}


def _cached_cm_data() -> str:
    if "default" not in _cm_data_cache:
        _cm_data_cache["default"] = _read_nri_configmap_data()
    return _cm_data_cache["default"]


def advanced_mode_active() -> bool:
    """Detect the advanced (multi-TP) balloon policy.

    Sources:
      1. MM_NRI_MODE=advanced env override.
      2. ConfigMap probe for 'vllm-balloon-tp1' substring.
    """
    mode = os.environ.get("MM_NRI_MODE", "").strip().lower()
    if mode == "advanced":
        return True
    if mode in ("generic", "single"):
        return False
    return "vllm-balloon-tp1" in _cached_cm_data()


def hyperthreads_hidden() -> bool:
    """Detect whether the NRI policy hides hyperthreads.

    Sources:
      1. MM_NRI_HIDE_HT env override.
      2. ConfigMap probe for 'hideHyperthreads: true'.
    """
    override = os.environ.get("MM_NRI_HIDE_HT", "").strip().lower()
    if override in ("1", "true", "yes"):
        return True
    if override in ("0", "false", "no"):
        return False
    return "hideHyperthreads: true" in _cached_cm_data()


# ── Balloon name resolution ─────────────────────────────────────────────────

def balloon_name(tp: int = 1, balloon_override: str = "") -> str:
    """Resolve balloon name using release-1 precedence.

    1. MM_NRI_BALLOON_NAME env override
    2. Explicit --balloon override
    3. Advanced mode → vllm-balloon-tp<N> (snapped to 1/2/4/8)
    4. Default → vllm-balloon
    """
    env_name = os.environ.get("MM_NRI_BALLOON_NAME", "").strip()
    if env_name:
        return env_name
    if balloon_override:
        return balloon_override
    if advanced_mode_active():
        for size in (1, 2, 4, 8):
            if tp <= size:
                return f"{_DEFAULT_BALLOON}-tp{size}"
        return f"{_DEFAULT_BALLOON}-tp8"
    return _DEFAULT_BALLOON


# ── Effective CPU ────────────────────────────────────────────────────────────

def effective_cpu_request(cpu: int, node: str = "") -> int:
    """Return the CPU request for the container.

    With the siblings reservation approach (default), all sibling (HT)
    cores are reserved at NRI install time. Balloons only allocate from
    physical cores, and hideHyperthreads is false. No doubling needed —
    the pod requests N CPUs and gets N physical cores.

    Legacy fallback: if hideHyperthreads is true (old-style config without
    sibling reservation), double the request.
    """
    if cpu <= 0:
        return cpu
    if hyperthreads_hidden():
        return cpu * 2
    return cpu


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
    annotations: dict[str, str] = {
        "cpu-policy.model-manager.io/type": "nri-balloons",
        "cpu-policy.model-manager.io/balloon": bname,
        _NRI_BALLOON_ANNOTATION: bname,
    }
    _info(f"Binding pods to NRI balloon '{bname}'")
    return annotations


# ── Hard node affinity (advanced NRI + TP>1) ─────────────────────────────────

def nri_node_selector_term(tp: int) -> dict | None:
    """Return a requiredDuringScheduling nodeSelectorTerm for NRI TP>1.

    In advanced mode, pods with TP>1 must land on nodes whose
    nri.intel.com/max-tp >= TP. Returns None when not applicable.
    """
    if tp <= 1:
        return None
    if not advanced_mode_active():
        return None
    return {
        "matchExpressions": [{
            "key": _NRI_MAX_TP_LABEL,
            "operator": "Gt",
            "values": [str(tp - 1)],
        }],
    }


# ── NUMA topology helpers ───────────────────────────────────────────────────

def _node_label(node: str, label: str) -> str:
    escaped = label.replace(".", "\\.")
    return _kubectl(
        "get", "node", node,
        "-o", f"jsonpath={{.metadata.labels.{escaped}}}",
        ignore_errors=True,
    )


def _node_allocatable_cpu(node: str) -> int:
    raw = _kubectl(
        "get", "node", node,
        "-o", "jsonpath={.status.allocatable.cpu}",
        ignore_errors=True,
    )
    return _cpu_to_cores(raw)


def _cpu_to_cores(val: str) -> int:
    """Convert CPU string (e.g. '96', '96000m') to whole cores."""
    val = val.strip()
    if not val:
        return 0
    if val.endswith("m"):
        try:
            return int(val[:-1]) // 1000
        except ValueError:
            return 0
    try:
        return int(val)
    except ValueError:
        try:
            return int(float(val))
        except ValueError:
            return 0


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


def _numa_nodes_count(node: str) -> int:
    """Read NUMA node count from node label."""
    raw = _node_label(node, "nri.intel.com/numa-nodes")
    try:
        return int(raw)
    except (ValueError, TypeError):
        return 0


def _physical_cpus(node: str) -> int:
    raw = _node_label(node, "nri.intel.com/physical-cpus")
    try:
        return int(raw)
    except (ValueError, TypeError):
        return 0


def _node_max_tp(node: str) -> int:
    raw = _node_label(node, _NRI_MAX_TP_LABEL)
    try:
        return int(raw) if raw else 0
    except (ValueError, TypeError):
        return 0


def _reserved_cpus_count(node: str) -> int:
    """Read reserved CPU count from the BalloonsPolicy CR."""
    policy_name = f"node.{node}"
    raw = _kubectl(
        "get", _BALLOONSPOLICY_CRD, policy_name,
        "-n", _NRI_BALLOONS_NS,
        "-o", "jsonpath={.spec.reservedResources.cpu}",
        ignore_errors=True,
    )
    if not raw:
        # Try default policy
        raw = _kubectl(
            "get", _BALLOONSPOLICY_CRD, "default",
            "-n", _NRI_BALLOONS_NS,
            "-o", "jsonpath={.spec.reservedResources.cpu}",
            ignore_errors=True,
        )
    if not raw:
        return 0
    if raw.startswith("cpuset:"):
        return len(_parse_cpuset(raw[7:]))
    try:
        return int(raw)
    except ValueError:
        return 0


def _balloon_pods_cpu_on_node(node: str) -> int:
    """Sum CPU requests of NRI-balloon-annotated pods on a node."""
    pods = _kubectl_json("get", "pods", "-A",
                         "--field-selector", f"spec.nodeName={node}")
    if not pods or "items" not in pods:
        return 0
    total_millis = 0
    for pod in pods["items"]:
        phase = pod.get("status", {}).get("phase", "")
        if phase not in ("Running", "Pending"):
            continue
        anns = pod.get("metadata", {}).get("annotations") or {}
        if not any(k.startswith(_NRI_ANNOTATION_PREFIX) for k in anns):
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

def _per_numa_capacity(node: str) -> list[int]:
    """Logical CPUs per NUMA node, from labels."""
    phys = _physical_cpus(node)
    numas = _numa_nodes_count(node)
    if phys <= 0 or numas <= 0:
        return []
    # Assume uniform. Total logical = allocatable CPU on node.
    alloc = _node_allocatable_cpu(node)
    if alloc <= 0:
        alloc = phys  # fallback
    # NRI works with physical cores, but logical CPUs are what the scheduler sees.
    # Use node capacity for logical count.
    cap_raw = _kubectl(
        "get", "node", node,
        "-o", "jsonpath={.status.capacity.cpu}",
        ignore_errors=True,
    )
    total_logical = _cpu_to_cores(cap_raw) if cap_raw else phys * 2
    per_numa = total_logical // numas
    return [per_numa] * numas


def _node_total_logical_cpu(node: str) -> int:
    """Total logical CPUs on a node from status.capacity.cpu (whole cores)."""
    raw = _kubectl(
        "get", "node", node,
        "-o", "jsonpath={.status.capacity.cpu}",
        ignore_errors=True,
    )
    return _cpu_to_cores(raw) if raw else 0


def _node_balloon_pool(node: str) -> int:
    """CPUs available to *named* balloons (vllm-balloon-tpN) on a node.

    This is the balloon-allocatable pool: total logical CPUs minus the
    reserved-pool seed. It is expressed in the same unit as
    `effective_cpu_request()` (a container's balloon footprint), so the two
    can be compared directly regardless of policy mode:

      - siblings profile (hideHyperthreads=false, all siblings reserved):
        reserved == sibling count, so the pool collapses to the physical-core
        count and effective_cpu == cpu. Comparing them enforces "physical
        cores only" without hardcoding a physical/sibling split.
      - legacy hideHyperthreads=true: reserved is small, the pool stays in
        logical CPUs, and effective_cpu == 2*cpu (the sibling is held idle),
        so the footprint accounting still balances.
      - partial-reserve profiles (standard/observability/full): the pool is
        total_logical minus the reserved seed; a plain oversubscription guard.

    Falls back to allocatable CPU when capacity is unreadable.
    """
    total = _node_total_logical_cpu(node)
    if total <= 0:
        total = _node_allocatable_cpu(node)
    reserved = _reserved_cpus_count(node)
    return max(0, total - reserved)


def _node_capacity_report(
    node: str, eff_cpu: int, tp: int, replicas: int = 1,
) -> tuple[bool, int, str]:
    """Evaluate whether a node can host `replicas` copies of the balloon.

    Works in the balloon-allocatable pool (see `_node_balloon_pool`) rather
    than raw allocatable/logical CPU, so the check reflects the CPUs NRI can
    actually hand to a `vllm-balloon-tpN` — the physical-core pool under the
    default siblings profile.

    Returns (fits_all_replicas, slots, detail_string) where `slots` is how
    many whole replicas fit on this node right now (used by the cluster-wide
    aggregate to model the scheduler packing replicas across nodes).
    """
    replicas = max(1, replicas)
    if eff_cpu <= 0:
        return (True, 0, f"{node}: eff_cpu<=0, capacity check skipped")

    pool = _node_balloon_pool(node)
    in_use = _balloon_pods_cpu_on_node(node)
    free = max(0, pool - in_use)

    # Node-wide slots: an upper bound on how many replicas the balloon pool can
    # hold ignoring NUMA fragmentation.
    node_wide_slots = free // eff_cpu

    numa_caps = _per_numa_capacity(node)
    if not numa_caps:
        # No NUMA topology labels — fall back to the node-wide pool budget.
        slots = node_wide_slots
        fits = eff_cpu * replicas <= free
        detail = (
            f"{node}: need {eff_cpu * replicas} ({eff_cpu} x {replicas}), "
            f"free {free} in balloon pool (pool={pool}, in-use={in_use}, "
            f"NUMA undetectable); node fits {slots} replica(s)"
        )
        return (fits, slots, detail)

    # NUMA-aware slot count. Each replica's TP shards are spread across `span`
    # NUMA nodes (balance-balloons), each shard needing `per_numa_need` CPUs
    # from that NUMA's share of the balloon pool. Existing balloon load is
    # assumed evenly spread across NUMA nodes (matches the allocator's
    # balancing and the prior model). A replica therefore consumes one
    # `per_numa_need`-sized "unit" from each of `span` NUMA nodes, so the node
    # holds floor(total_units / span) replicas — this naturally caps the
    # node-wide estimate when cores don't divide evenly per NUMA.
    reserved = _reserved_cpus_count(node)
    reserved_per_numa = reserved // len(numa_caps)
    per_numa_used = in_use // len(numa_caps)
    usable = [max(0, c - reserved_per_numa - per_numa_used) for c in numa_caps]

    span = max(1, min(tp, len(usable)))
    per_numa_need = math.ceil(eff_cpu / span)

    numa_ge_need = sum(1 for u in usable if u >= per_numa_need)
    if numa_ge_need < span:
        # A single replica cannot even place its shards — node hosts zero.
        return (False, 0,
                f"{node}: tp={tp} needs {span} NUMA node(s) with >= "
                f"{per_numa_need} balloon CPU each, only {numa_ge_need} qualify "
                f"(per-NUMA balloon pool free: {usable})")

    total_units = sum(u // per_numa_need for u in usable)
    numa_slots = total_units // span
    slots = min(node_wide_slots, numa_slots)

    fits = slots >= replicas
    detail = (
        f"{node}: need {replicas} replica(s) of {eff_cpu} (tp={tp}, "
        f"{per_numa_need}/NUMA x {span} NUMA); node fits {slots} "
        f"(pool={pool}, in-use={in_use}, per-NUMA free={usable})"
    )
    return (fits, slots, detail)


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

    eff_cpu = effective_cpu_request(cpu, node)
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

    # ── NUMA topology checks ──
    target = node
    if not target:
        # Pick first NRI-capable node for topology checks
        candidates = _schedulable_nri_nodes(tp)
        if candidates:
            target = candidates[0]

    if target:
        numas = _numa_nodes_count(target)
        phys = _physical_cpus(target)

        if numas <= 0:
            notes.append(f"NUMA topology not detectable for '{target}'; skipping NUMA checks")
        else:
            if tp > numas:
                errors.append(
                    f"TP={tp} exceeds NUMA node count {numas} on '{target}' — "
                    f"balloon '{bname}' cannot honour NUMA isolation"
                )
            cores_per_numa = phys // numas
            if cores_per_numa > 0 and cpu > cores_per_numa * tp:
                errors.append(
                    f"physical CPU request {cpu} > {cores_per_numa * tp} "
                    f"(cores_per_numa={cores_per_numa} x tp={tp}) on '{target}'"
                )

    # ── Capacity fit: balloon-pool budget + per-NUMA shard placement ──
    #
    # The check runs in the balloon-allocatable pool (total logical CPUs minus
    # the reserved seed). Under the default 'siblings' profile that pool equals
    # the physical-core count, so this enforces "vLLM gets physical cores only"
    # — a model whose replicas would overflow the physical pool is blocked here
    # instead of silently spilling onto reserved sibling cores at runtime.
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


# ── CLI entry point ──────────────────────────────────────────────────────────

def cmd_detect(args: argparse.Namespace) -> None:
    policy = detect_policy(node=args.node or "")
    result = {
        "policy": policy,
        "nri_installed": _nri_balloons_installed(),
        "advanced_mode": advanced_mode_active(),
        "hide_hyperthreads": hyperthreads_hidden(),
    }
    print(json.dumps(result))


def cmd_resolve(args: argparse.Namespace) -> None:
    detect_policy(node=args.node or "")
    cpu = int(args.cpu or 0)
    tp = int(args.tp or 1)
    eff_cpu = effective_cpu_request(cpu, args.node or "")
    bname = balloon_name(tp, args.balloon or "")
    anns = build_annotations(tp, args.kind or "LLMInferenceService", args.balloon or "")
    nst = nri_node_selector_term(tp)

    result = {
        "policy": "nri-balloons",
        "balloon": bname,
        "effective_cpu": eff_cpu,
        "annotations": anns,
        "node_selector_term": nst,
        "advanced_mode": advanced_mode_active(),
        "hide_hyperthreads": hyperthreads_hidden(),
    }
    print(json.dumps(result))


def cmd_preflight(args: argparse.Namespace) -> None:
    result = preflight(
        cpu=int(args.cpu or 0),
        tp=int(args.tp or 1),
        node=args.node or "",
        replicas=int(args.replicas or 1),
    )
    print(json.dumps(result))
    if not result["ok"]:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="NRI balloon policy helper")
    sub = parser.add_subparsers(dest="command")

    p_detect = sub.add_parser("detect", help="Verify NRI balloons status")
    p_detect.add_argument("--node", default="")

    p_resolve = sub.add_parser("resolve", help="Resolve NRI settings for a deployment")
    p_resolve.add_argument("--cpu", required=True)
    p_resolve.add_argument("--tp", default="1")
    p_resolve.add_argument("--node", default="")
    p_resolve.add_argument("--kind", default="LLMInferenceService")
    p_resolve.add_argument("--balloon", default="")

    p_preflight = sub.add_parser("preflight", help="Pre-deploy NRI validation")
    p_preflight.add_argument("--cpu", required=True)
    p_preflight.add_argument("--tp", default="1")
    p_preflight.add_argument("--node", default="")
    p_preflight.add_argument("--replicas", default="1")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "detect":
        cmd_detect(args)
    elif args.command == "resolve":
        cmd_resolve(args)
    elif args.command == "preflight":
        cmd_preflight(args)


if __name__ == "__main__":
    main()
