#!/usr/bin/env python3
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Report NRI balloons CPU assignment for vLLM pods.

Shows: Pod | Balloon | Node | NUMA Node | CPUs | Siblings

Usage:
    python3 scripts/nri-cpu-report.py [--namespace llm-inference] [--kubeconfig PATH]

Requires: kubernetes Python client (pip install kubernetes)
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

try:
    from kubernetes import client, config
    from kubernetes.stream import stream
except ImportError:
    sys.exit("ERROR: 'kubernetes' package not installed. Run: pip install kubernetes")


_NRI_ANN_PREFIX = "balloon.balloons.resource-policy.nri.io/"
_POD_LABEL_PREFIX = "pod/labels/"


def _label_balloon_types(custom_api) -> list[tuple[str, list[dict]]]:
    """Balloon types that select pods by label (matchExpressions on pod/labels/...), from the
    BalloonsPolicy. OpenShift AI's v1alpha1 LLMInferenceService cannot annotate its pods, so they
    land in their balloon this way. Node-specific policies repeat the default's types."""
    try:
        policies = custom_api.list_cluster_custom_object("config.nri", "v1alpha1", "balloonspolicies")
    except Exception:
        return []
    types, seen = [], set()
    for policy in policies.get("items", []):
        for bt in (policy.get("spec") or {}).get("balloonTypes") or []:
            exprs = bt.get("matchExpressions") or []
            if bt["name"] in seen or not exprs or not all(e.get("key", "").startswith(_POD_LABEL_PREFIX)
                                                          for e in exprs):
                continue
            seen.add(bt["name"])
            types.append((bt["name"], exprs))
    return types


def _matches(labels: dict, exprs: list[dict]) -> bool:
    for e in exprs:
        value = labels.get(e["key"][len(_POD_LABEL_PREFIX):])
        op, values = e.get("operator"), e.get("values") or []
        ok = {"In": value in values, "NotIn": value not in values, "Equals": value == (values or [None])[0],
              "NotEqual": value != (values or [None])[0], "Exists": value is not None,
              "NotExist": value is None}.get(op, False)
        if not ok:
            return False
    return True


def _parse_cpuset(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _cpuset_to_str(cpus: list[int]) -> str:
    if not cpus:
        return ""
    ranges = []
    start = prev = cpus[0]
    for c in cpus[1:]:
        if c == prev + 1:
            prev = c
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = c
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def _exec_in_pod(core_api, pod_name: str, namespace: str, cmd: list[str],
                 container: str = "main") -> str:
    try:
        return stream(
            core_api.connect_get_namespaced_pod_exec,
            pod_name, namespace, container=container,
            command=cmd, stderr=True, stdin=False, stdout=True, tty=False,
        )
    except Exception:
        try:
            return stream(
                core_api.connect_get_namespaced_pod_exec,
                pod_name, namespace,
                command=cmd, stderr=True, stdin=False, stdout=True, tty=False,
            )
        except Exception:
            return ""


def _get_topology(core_api, pod_name: str, namespace: str):
    """Get sibling map and numa map by exec'ing into a pod on the node."""
    raw = _exec_in_pod(core_api, pod_name, namespace, [
        "sh", "-c",
        "for f in /sys/devices/system/cpu/cpu*/topology/thread_siblings_list; do "
        "cpu=$(echo $f | grep -o 'cpu[0-9]*' | grep -o '[0-9]*'); "
        "echo \"$cpu $(cat $f)\"; done"
    ])
    sibling_map: dict[int, int] = {}
    for line in raw.strip().splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        cpu_id = int(parts[0])
        siblings = _parse_cpuset(parts[1])
        # Map every CPU to the physical core = lowest CPU in its sibling group.
        # This way both threads of a core map to the same key.
        phys_core = min(siblings) if siblings else cpu_id
        sibling_map[cpu_id] = phys_core

    raw = _exec_in_pod(core_api, pod_name, namespace, [
        "sh", "-c",
        "for d in /sys/devices/system/node/node*; do "
        "node=$(basename $d | grep -o '[0-9]*'); "
        "cpus=$(cat $d/cpulist 2>/dev/null); "
        "echo \"$node $cpus\"; done"
    ])
    numa_map: dict[int, int] = {}
    for line in raw.strip().splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        numa_id = int(parts[0])
        for cpu in _parse_cpuset(parts[1]):
            numa_map[cpu] = numa_id

    return sibling_map, numa_map


def main():
    parser = argparse.ArgumentParser(description="NRI balloons CPU assignment report")
    parser.add_argument("-n", "--namespace", default="llm-inference")
    parser.add_argument("--kubeconfig", default=None)
    args = parser.parse_args()

    try:
        if args.kubeconfig:
            config.load_kube_config(config_file=args.kubeconfig)
        else:
            try:
                config.load_kube_config()
            except Exception:
                config.load_incluster_config()
    except Exception as e:
        sys.exit(f"ERROR: Cannot load kubeconfig: {e}")

    core_api = client.CoreV1Api()
    pods = core_api.list_namespaced_pod(args.namespace)
    label_types = _label_balloon_types(client.CustomObjectsApi())

    balloon_pods = []
    for pod in pods.items:
        if pod.status.phase != "Running":
            continue
        anns = pod.metadata.annotations or {}
        balloon = next((v for k, v in anns.items() if k.startswith(_NRI_ANN_PREFIX)), None)
        if not balloon:
            balloon = next((name for name, exprs in label_types
                            if _matches(pod.metadata.labels or {}, exprs)), None)
        if balloon:
            balloon_pods.append({
                "name": pod.metadata.name,
                "node": pod.spec.node_name,
                "balloon": balloon,
            })

    if not balloon_pods:
        print(f"No running pods with NRI balloon annotations or balloon-matched labels in '{args.namespace}'")
        return

    # Get topology from first pod on each node
    topo_cache: dict[str, tuple] = {}
    for p in balloon_pods:
        if p["node"] not in topo_cache:
            topo_cache[p["node"]] = _get_topology(core_api, p["name"], args.namespace)

    # Get cpuset for each pod
    for p in balloon_pods:
        raw = _exec_in_pod(core_api, p["name"], args.namespace, ["cat", "/proc/self/status"])
        cpus = []
        for line in raw.splitlines():
            if line.startswith("Cpus_allowed_list:"):
                cpus = _parse_cpuset(line.split(":", 1)[1])
                break
        p["cpus"] = cpus

    # Print table
    print()
    hdr = f"{'Pod':<48} {'Balloon':<20} {'Node':<12} {'NUMA':<8} {'CPUs':<24} {'Siblings':<24} {'HT Mode'}"
    print(hdr)
    print("-" * len(hdr))

    for p in balloon_pods:
        smap, nmap = topo_cache.get(p["node"], ({}, {}))
        cpus = p["cpus"]

        # Determine NUMA node(s)
        numas = sorted(set(nmap.get(c, -1) for c in cpus))
        numa_str = ",".join(str(n) for n in numas if n >= 0) or "?"

        # Compute siblings (HT partners not in this pod's cpuset)
        # smap maps cpu → physical_core (lowest sibling). Two CPUs sharing the
        # same physical core are siblings.
        phys_to_cpus: dict[int, list[int]] = defaultdict(list)
        for c in cpus:
            phys_to_cpus[smap.get(c, c)].append(c)
        siblings = sorted(
            s for c in cpus
            for s in [s2 for s2, p in smap.items() if p == smap.get(c, c)]
            if s not in set(cpus)
        )
        sib_str = _cpuset_to_str(siblings) if siblings else "-"

        # Determine HT mode: does this pod use both threads of any physical core?
        cpus_set = set(cpus)
        both_threads = 0
        phys_only = 0
        for phys, members in phys_to_cpus.items():
            # Check if ALL threads of this core are in the pod's cpuset
            all_threads = [s for s, p in smap.items() if p == phys]
            if len(all_threads) > 1 and cpus_set.issuperset(all_threads):
                both_threads += 1
            else:
                phys_only += 1

        if both_threads == 0:
            ht_mode = f"physical only ({phys_only}c)"
        elif phys_only == 0:
            ht_mode = f"phys+sibling ({both_threads}c×2t)"
        else:
            ht_mode = f"mixed ({phys_only}c + {both_threads}c×2t)"

        cpu_str = _cpuset_to_str(cpus) if cpus else "(unavailable)"
        print(f"{p['name']:<48} {p['balloon']:<20} {p['node']:<12} {numa_str:<8} {cpu_str:<24} {sib_str:<24} {ht_mode}")

    # Collision check
    pods_by_node: dict[str, list] = defaultdict(list)
    for p in balloon_pods:
        pods_by_node[p["node"]].append(p)

    collisions = []
    for node, node_pods in pods_by_node.items():
        if len(node_pods) < 2:
            continue
        smap, _ = topo_cache.get(node, ({}, {}))
        core_to_pods: dict[int, list[str]] = defaultdict(list)
        unmapped_cpus: list[tuple[str, int]] = []
        for p in node_pods:
            for cpu in p["cpus"]:
                if cpu not in smap:
                    unmapped_cpus.append((p["name"], cpu))
                    continue
                phys = smap[cpu]  # lowest sibling = physical core id
                if p["name"] not in core_to_pods[phys]:
                    core_to_pods[phys].append(p["name"])
        for phys, names in sorted(core_to_pods.items()):
            if len(names) > 1:
                # Show all CPUs that map to this physical core
                threads = sorted(c for c, p in smap.items() if p == phys)
                collisions.append((phys, _cpuset_to_str(threads), names))
        if unmapped_cpus:
            print(f"  WARNING: {node}: {len(unmapped_cpus)} CPU(s) missing from topology map (collision detection incomplete):")
            for pod_name, cpu_id in unmapped_cpus[:10]:
                print(f"    cpu {cpu_id} in pod {pod_name}")
            if len(unmapped_cpus) > 10:
                print(f"    ... and {len(unmapped_cpus) - 10} more")

    print()
    if collisions:
        print("COLLISIONS:")
        print(f"{'Phys Core':<12} {'CPU IDs':<20} {'Pods'}")
        print("-" * 60)
        for phys, ids, names in collisions:
            print(f"{phys:<12} {ids:<20} {', '.join(names)}")
    else:
        print("No collisions — each physical core used by at most one pod.")
    print()


if __name__ == "__main__":
    main()
