"""GPU runtime (vllm-cuda): parity cases and unit tests.

Kept apart from test_parity.py and test_units.py so that additions there don't
collide with these. The parity cases reuse test_parity's harness. Their goldens
(tests/golden/gpu-*.txt) come from the legacy GPU branch (DCAIDSE8-2342, commit
0d38ae1), which served Llama 3.1 70B on 8x NVIDIA B300 and reproduces every
other golden; regenerate them with `python tests/test_gpu.py <legacy model_manager
dir>` from that branch, with bash 5.2's patsub_replacement off.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import cli, config, download, k8s, policy, renderer  # noqa: E402
from core.config import Catalog, Model  # noqa: E402
from test_parity import (BE, DOWNLOAD, GOLDEN, HF_EXCLUDE_DEV, MM, NO_LITELLM_URL, NRI, POOL,  # noqa: E402
                         _ready, case, run)
from test_units import _quiet_deploy  # noqa: E402

# GPU pods are Burstable since the B300 run: no CPU limit, so kubelet's static CPU manager never pins them.
BURSTABLE_GPU = (r'^(          limits:)\n            cpu: "[^"]+"\n(            memory: \S+\n            nvidia\.com/gpu:)',
                 r"\1\n\2")
# The weight-download Job tolerates the GPU taint too, so it can run where only tainted GPU nodes fit.
GPU_DOWNLOAD_TOLERATION = (r"^(      restartPolicy: Never\n)",
                           r"\1      tolerations:\n        - key: nvidia.com/gpu\n          operator: Exists\n"
                           r"          effect: NoSchedule\n")
# The download Job requests no GPU, so it is shown none (the CUDA image sets NVIDIA_VISIBLE_DEVICES=all).
GPU_DOWNLOAD_NO_GPUS = (r'^(            - name: HF_HOME\n              value: "/models/\.hf-cache"\n)',
                        r'\1            - name: NVIDIA_VISIBLE_DEVICES\n              value: "void"\n')
# No gpu-vendor node preference: the accelerator request alone places a GPU pod.
VENDOR_PREFERENCE = (r"          - weight: 100\n            preference:\n              matchExpressions:\n"
                     r"                - key: gpu-vendor\n                  operator: In\n                  values:\n"
                     r"                    - nvidia\n")
NO_VENDOR_PREFERENCE = (r"^    affinity:\n      nodeAffinity:\n        preferredDuringSchedulingIgnoredDuringExecution:\n"
                        + VENDOR_PREFERENCE, "")
# No deploy gate: a --dry-run without MM_ACCEL_RESOURCE previews without a warning.
NO_GATE_WARNING = (r"^\[WARN \]  MM_ACCEL_RESOURCE is not set .*\n", "")
# --tp without --gpus sets the GPU count too, so a TP 4 pod no longer holds 8 GPUs.
TP_SETS_GPUS = [(r"^(\[INFO \]    GPUs: )8( x nvidia\.com/gpu)$", r"\g<1>4\2"), (r'^( +nvidia\.com/gpu: )"8"$', r'\1"4"')]
LLAMA_GPU_READY = [[r"get LLMInferenceService llama-3-1-70b-instruct-cuda -n llm-inference -o json$", 0,
                    _ready("MainWorkloadReady")]]


def _nodes(*gpus, unschedulable=(), not_ready=(), taints=None):
    """`kubectl get nodes -o json`: node n<i> with that many allocatable GPUs (None: no GPUs)."""
    return {"items": [{"metadata": {"name": f"n{i}"},
                       "spec": {"unschedulable": i in unschedulable, "taints": (taints or {}).get(i, [])},
                       "status": {"allocatable": {"cpu": "64", "memory": "1Ti", **({} if g is None else {"nvidia.com/gpu": str(g)})},
                                  "conditions": [{"type": "Ready", "status": "False" if i in not_ready else "True"}]}}
                      for i, g in enumerate(gpus)]}


def _pods(*held):
    """`kubectl get pods -A -o json`: (node, model name, GPUs requested) per pod, in namespace n."""
    return {"items": [{"metadata": {"namespace": "n", "labels": {"app.kubernetes.io/name": name}},
                       "spec": {"nodeName": node, "containers": [{"resources": {"requests": {"nvidia.com/gpu": str(n)}}}]}}
                      for node, name, n in held]}


# --node is a required affinity since #74 (it was preferred, with a control-plane preference), and there is no GPU
# vendor preference.
PIN_NODE_GPU = (r"^        preferredDuringSchedulingIgnoredDuringExecution:\n          - weight: 100\n            preference:\n"
                r"              matchExpressions:\n                - key: kubernetes\.io/hostname\n                  operator: In\n"
                r"                  values:\n                    - w1\n          - weight: 50\n[\s\S]*?operator: DoesNotExist\n"
                + VENDOR_PREFERENCE,
                "        requiredDuringSchedulingIgnoredDuringExecution:\n          nodeSelectorTerms:\n            - matchExpressions:\n"
                "                - key: kubernetes.io/hostname\n                  operator: In\n                  values:\n"
                '                    - "w1"\n')
GPU_NODES = [[r"get nodes -o json$", 0, json.dumps(_nodes(8))],
             [r"get pods -A --field-selector \S+ -o json$", 0, json.dumps(_pods())]]

GPU_CASES = [
    # gpu-llama-70b is the manifest that served on B300.
    case("gpu-llama-70b", "deploy", "llama-3-1-70b-instruct-cuda", "--dry-run",
         deviations=[BURSTABLE_GPU, NO_VENDOR_PREFERENCE]),
    case("gpu-dry-run", "deploy", "qwen3-0-6b-cuda", "--dry-run", deviations=[BURSTABLE_GPU, NO_GATE_WARNING]),
    case("gpu-count-node", "deploy", "qwen3-0-6b-cuda", "--gpus", "2", "--node", "w1", "--dry-run", env=NRI,
         deviations=[BURSTABLE_GPU, PIN_NODE_GPU]),
    case("gpu-explicit-tp", "deploy", "llama-3-1-70b-instruct-cuda", "--tp", "4", "--dry-run",
         deviations=[BURSTABLE_GPU, *TP_SETS_GPUS, NO_VENDOR_PREFERENCE]),
    case("gpu-flag-ignored-on-cpu", "deploy", "qwen3-0-6b", "--gpus", "2", "--dry-run", env=BE),
    case("gpu-bad-count", "deploy", "qwen3-0-6b-cuda", "--gpus", "x"),
    case("gpu-deploy-wait", "deploy", "llama-3-1-70b-instruct-cuda", "--wait",
         k8s=GPU_NODES + DOWNLOAD + POOL + LLAMA_GPU_READY,
         deviations=[NO_LITELLM_URL, BURSTABLE_GPU, GPU_DOWNLOAD_TOLERATION, GPU_DOWNLOAD_NO_GPUS, NO_VENDOR_PREFERENCE]),
]


@pytest.mark.parametrize("c", GPU_CASES)
def test_gpu_parity(c, tmp_path, request):
    expected = (GOLDEN / f"{request.node.callspec.id}.txt").read_text()
    for pattern, repl in c["deviations"]:
        expected, n = re.subn(pattern, repl, expected, flags=re.M)
        assert n, f"stale deviation: {pattern}"
    expected, n = re.subn(*HF_EXCLUDE_DEV, expected, flags=re.M)
    assert n == len(re.findall(r"^\s+hf download '", expected, re.M)), "stale deviation: HF_EXCLUDE_DEV"
    assert run(MM / "model-manager", c, tmp_path) == expected


def test_zero_gpus_rejected():
    with pytest.raises(SystemExit):
        cli.parse_args(["deploy", "m", "--gpus", "0"])


def test_gpu_tp_defaults_to_gpu_count():
    assert cli._engine_cli_args(cli.Options(), 8, []) == (["--tensor-parallel-size", "8"], 8)
    assert cli._engine_cli_args(cli.Options(), 1, []) == ([], 1)
    assert cli._engine_cli_args(cli.Options(tp="4"), 8, []) == (["--tensor-parallel-size", "4"], 4)


def test_catalog_tp_is_not_overridden():
    # A tensor-parallel size in the catalog (category, version or model args) stands; the GPU-count
    # default would render after it and win, since dedupe_args keeps the last occurrence.
    assert cli._engine_cli_args(cli.Options(), 8, ["--tensor-parallel-size", "4"]) == ([], 4)
    assert cli._engine_cli_args(cli.Options(), 8, ["--max-model-len", "8192", "--tensor-parallel-size=2"]) == ([], 2)
    assert cli._engine_cli_args(cli.Options(), 8, ["--max-model-len", "8192"]) == (["--tensor-parallel-size", "8"], 8)
    # --tp on the command line still overrides the catalog.
    assert cli._engine_cli_args(cli.Options(tp="8"), 8, ["--tensor-parallel-size", "4"]) == (
        ["--tensor-parallel-size", "8"], 8)
    args, _ = cli._engine_cli_args(cli.Options(), 8, ["--tensor-parallel-size", "4"])
    assert renderer.dedupe_args(["--tensor-parallel-size", "4", *args]) == ["--tensor-parallel-size", "4"]
    with pytest.raises(SystemExit):
        cli._engine_cli_args(cli.Options(), 8, ["--tensor-parallel-size", "x"])


def test_a_runtime_requests_its_own_resource(monkeypatch):
    # Each GPU runtime asks for its own accelerator, so a cluster with another vendor's GPUs refuses it.
    xe = {**GPU_MODEL["rt"], "accelerator_resource": "gpu.intel.com/xe", "accelerator_toleration_key": ""}
    assert Model(name="m", namespace="n", rt=xe).accel_resource == "gpu.intel.com/xe"
    assert preflight(monkeypatch, _nodes(8), gpus="1", rt=xe) == [
        "No Ready, schedulable node advertises allocatable gpu.intel.com/xe: no GPU node has joined the cluster, "
        "or its device plugin is not ready yet."]
    assert preflight(monkeypatch, _nodes(8), gpus="1") == []  # the NVIDIA runtime fits the same nodes


def test_gpu_runtime_only_as_llm_inference_service():
    m = Model(name="m", namespace="n", runtime="vllm-cuda", gpus="2", kind="InferenceService",
              rt={"device": "gpu", "accelerator_resource": "nvidia.com/gpu"})
    with pytest.raises(SystemExit):
        cli._check_gpu(m)


def test_a_gpu_runtime_must_name_its_resource(capsys):
    with pytest.raises(SystemExit):
        cli._check_gpu(Model(name="m", namespace="n", runtime="x", gpus="1", rt={"device": "gpu"}))
    assert "names no accelerator_resource" in capsys.readouterr().err


def test_free_gpus_are_never_reported_negative(monkeypatch):
    assert "the most on one node is 0" in preflight(monkeypatch, _nodes(4), _pods(("n0", "other", 8)))[0]


def test_a_complete_gpu_runtime_passes_the_check():
    m = Model(name="m", namespace="n", runtime="vllm-cuda", gpus="1",
              rt={"device": "gpu", "accelerator_resource": "nvidia.com/gpu"})
    cli._check_gpu(m)
    assert m.gpu and m.accel_resource == "nvidia.com/gpu"
    assert not Model(name="m", namespace="n", rt={}).gpu


@pytest.mark.parametrize("gpus", ["0", "two", ""])
def test_invalid_catalog_gpu_count_refused(gpus):
    # --gpus is checked by the parser; a catalog or category value only reaches this check.
    m = Model(name="m", namespace="n", runtime="vllm-cuda", gpus=gpus,
              rt={"device": "gpu", "accelerator_resource": "nvidia.com/gpu"})
    with pytest.raises(SystemExit):
        cli._check_gpu(m)


def test_check_gpu_is_inert_on_cpu():
    cli._check_gpu(Model(name="m", namespace="n", runtime="vllm", rt={"device": "cpu"}))


GPU_MODEL = {"name": "m", "namespace": "n", "runtime": "vllm-cuda",
             "rt": {"device": "gpu", "accelerator_resource": "nvidia.com/gpu", "accelerator_toleration_key": "nvidia.com/gpu"}}
GPU_TAINT = {"key": "nvidia.com/gpu", "effect": "NoSchedule"}


def preflight(monkeypatch, nodes, pods=None, pvc=None, pv=None, binding="", rolls=True, runtime=None, **model):
    """GpuRuntime.preflight against a fake cluster: nodes, pods, the model-store PVC and its PV."""
    objects = {"nodes": nodes, "pods": pods or _pods(), "pvc": pvc, "pv": pv}
    monkeypatch.setattr(k8s, "get_json", lambda *a: objects[a[0]])
    monkeypatch.setattr(k8s, "out", lambda *a: binding if a[:2] == ("get", "storageclass") else "")
    runtime = runtime or policy.GpuRuntime("model-store")
    runtime.rolls = rolls
    return runtime.preflight(Model(**{**GPU_MODEL, "gpus": "8", **model}), 1)


BOUND = {"spec": {"volumeName": "pv-1", "storageClassName": "local-path"}}
PENDING = {"spec": {"storageClassName": "local-path"}}


def _pv(*hosts):
    """A local-path PV pinned to these node names; none: shared storage."""
    if not hosts:
        return {"spec": {}}
    term = {"matchExpressions": [{"key": "kubernetes.io/hostname", "operator": "In", "values": list(hosts)}]}
    return {"spec": {"nodeAffinity": {"required": {"nodeSelectorTerms": [term]}}}}


def _labelled(nodes):
    for n in nodes["items"]:
        n["metadata"]["labels"] = {"kubernetes.io/hostname": n["metadata"]["name"]}
    return nodes


@pytest.mark.parametrize("nodes, pods, gpus, fits", [
    (_nodes(None, None), None, "1", False),                       # no GPU node has joined
    (_nodes(4, 4), None, "8", False),                             # a pod's GPUs cannot span nodes
    (_nodes(None, 8), None, "8", True),
    (_nodes(8, 2, unschedulable={0}), None, "4", False),          # cordoned
    (_nodes(8, not_ready={0}), None, "1", False),                 # NotReady, though allocatable is still reported
    (_nodes(8, taints={0: [{"key": "maintenance", "effect": "NoSchedule"}]}), None, "1", False),
    (_nodes(8, taints={0: [GPU_TAINT]}), None, "8", True),        # the GPU taint is tolerated
    (_nodes(8), _pods(("n0", "other", 8)), "1", False),           # all held by another model
    (_nodes(8), _pods(("n0", "m", 8)), "8", False),               # its own: a changed re-deploy rolls a new pod
    (_nodes(8, 8), _pods(("n0", "other", 8)), "8", True),         # the other node is free
])
def test_gpu_preflight_counts_free_gpus(monkeypatch, nodes, pods, gpus, fits):
    assert (preflight(monkeypatch, nodes, pods, gpus=gpus) == []) is fits


def test_gpu_preflight_messages(monkeypatch):
    assert preflight(monkeypatch, _nodes(None)) == [
        "No Ready, schedulable node advertises allocatable nvidia.com/gpu: no GPU node has joined the cluster, "
        "or its device plugin is not ready yet."]
    assert preflight(monkeypatch, _nodes(4, 2)) == [
        "No node has 8 free nvidia.com/gpu; the most on one node is 4. A pod's GPUs cannot span nodes: lower "
        "--gpus, or free GPUs on a node."]


def test_own_running_pods_name_the_remedy(monkeypatch):
    assert preflight(monkeypatch, _nodes(8), _pods(("n0", "m", 8))) == [
        "m's own running pods hold the nvidia.com/gpu it needs: if this change restarts the pod (any change but "
        "--replicas may), the rollout starts the new one next to the running one. Undeploy it first (./model-manager "
        "undeploy m), or use MM_PREFLIGHT=warn to apply anyway (e.g. a label-only change)."]
    # Same name in another namespace is another model.
    other_ns = {"items": [{**p, "metadata": {**p["metadata"], "namespace": "x"}} for p in _pods(("n0", "m", 8))["items"]]}
    assert preflight(monkeypatch, _nodes(8), other_ns)[0].startswith("No node has 8 free")


@pytest.mark.parametrize("pvc, pv, binding, fits, download", [
    (BOUND, _pv("cpu1"), "", False, ()),                       # pinned to a CPU worker: the GPU pod could never mount it
    (BOUND, _pv("n0"), "", True, ()),                          # pinned to the GPU node
    (PENDING, None, "WaitForFirstConsumer", True, ("n0",)),    # binds where the download Job runs: pin the Job
    (BOUND, _pv(), "", True, ()),                              # shared storage, no node affinity
    (PENDING, None, "Immediate", True, ()),
    ({"spec": {**PENDING["spec"], "accessModes": ["ReadWriteMany"]}}, None, "WaitForFirstConsumer", True, ()),  # shared
])
def test_gpu_preflight_and_the_model_store(monkeypatch, pvc, pv, binding, fits, download):
    nodes = _labelled({"items": _nodes(8)["items"] + _nodes(None)["items"]})
    nodes["items"][1]["metadata"] = {"name": "cpu1", "labels": {"kubernetes.io/hostname": "cpu1"}}
    runtime = policy.GpuRuntime("model-store")
    errors = preflight(monkeypatch, nodes, pvc=pvc, pv=pv, binding=binding, runtime=runtime)
    assert (errors == []) is fits and runtime.download_nodes == download
    if not fits:
        assert "GPU models need shared (ReadWriteMany) storage, or the model store on a GPU node." in errors[0]


def test_a_store_that_binds_on_first_use_is_created_where_the_replicas_fit(monkeypatch):
    # n1 holds one 4-GPU replica, n0 both: the Job must not land the store on n1.
    runtime = policy.GpuRuntime("model-store")
    assert preflight(monkeypatch, _labelled(_nodes(8, 4)), pvc=PENDING, binding="WaitForFirstConsumer",
                     gpus="4", replicas="2", runtime=runtime) == []
    assert runtime.download_nodes == ("n0",)


def test_an_unreadable_store_volume_is_warned_about(monkeypatch, capsys):
    assert preflight(monkeypatch, _labelled(_nodes(8)), pvc=BOUND, pv=None) == []
    assert "Could not read PersistentVolume pv-1 — skipping the model-store node check." in capsys.readouterr().err


def test_own_pods_on_the_stores_node_say_undeploy_not_storage(monkeypatch):
    nodes = _labelled(_nodes(8, 8))
    errors = preflight(monkeypatch, nodes, _pods(("n0", "m", 8)), pvc=BOUND, pv=_pv("n0"))
    assert errors[0].startswith("m's own running pods hold")


def test_a_store_that_binds_on_first_use_holds_replicas_on_one_node(monkeypatch, capsys):
    runtime = policy.GpuRuntime("model-store")
    errors = preflight(monkeypatch, _labelled(_nodes(8, 8)), pvc=PENDING, binding="WaitForFirstConsumer",
                       replicas="2", runtime=runtime)
    assert errors[0].startswith("Only 1 of 2 new replicas fit on the free nvidia.com/gpu of one node")
    assert runtime.download_nodes == ("n0", "n1")  # pinned even so, for MM_PREFLIGHT=warn
    assert "every model using it, CPU ones included, must then run on that node" in capsys.readouterr().err


@pytest.mark.parametrize("init, sidecars, app, held", [
    ([8], [], [0], 8),        # an init container's GPUs are reserved while it runs
    ([], [2], [4], 6),        # a restartable sidecar runs next to the app containers
    ([4], [2], [1], 6),       # init peak: the sidecar started before it, plus the init container
])
def test_a_pods_effective_gpu_request(init, sidecars, app, held):
    gpu = lambda n: {"resources": {"requests": {"nvidia.com/gpu": str(n)}}}  # noqa: E731
    pod = {"spec": {"initContainers": [{**gpu(n), "restartPolicy": "Always"} for n in sidecars] + [gpu(n) for n in init],
                    "containers": [gpu(n) for n in app]}}
    assert policy._pod_request(pod, "nvidia.com/gpu") == held


@pytest.mark.parametrize("nodes, pods, replicas, fits", [
    (_nodes(8, 8), _pods(("n0", "m", 8), ("n1", "m", 8)), "1", True),   # a scale-down releases GPUs
    (_nodes(8, 8, 8), _pods(("n0", "m", 8), ("n1", "m", 8)), "3", True),  # a scale-up needs room for the new replica only
    (_nodes(8, 8), _pods(("n0", "m", 8), ("n1", "m", 8)), "3", False),  # ... and refuses when there is none
])
def test_a_replica_only_change_needs_no_room_for_the_running_pods(monkeypatch, nodes, pods, replicas, fits):
    assert (preflight(monkeypatch, nodes, pods, rolls=False, replicas=replicas) == []) is fits


@pytest.mark.parametrize("nodes, gpus, fits", [
    (_nodes(5), "2", False),      # three 1-GPU replicas to 2 GPUs each: one surge pod fits, 3 x 2 never does
    (_nodes(4), "2", False),      # the node is full, and undeploying would not help either
    (_nodes(6), "2", True),       # the end state fits
    (_nodes(5), "1", True),       # same size
])
def test_a_rollout_whose_end_state_cannot_fit_is_refused(monkeypatch, nodes, gpus, fits):
    errors = preflight(monkeypatch, nodes, _pods(*[("n0", "m", 1)] * 3), gpus=gpus, replicas="3")
    assert (errors == []) is fits
    if not fits:
        assert errors == ["Once its running pods are replaced, only 2 of 3 replicas fit, so the rollout could not "
                          "finish: lower --gpus or --replicas, or free nvidia.com/gpu."]


def test_the_end_state_refusal_names_the_pinned_node(monkeypatch):
    # Three 1-GPU replicas on n1, moved with --node to n0, which has room for 2.
    errors = preflight(monkeypatch, _nodes(2, 8), _pods(*[("n1", "m", 1)] * 3), gpus="1", replicas="3", node="n0")
    assert errors[0].startswith("Once its running pods are replaced, only 2 of 3 replicas fit on n0,")


def test_a_pending_replica_still_needs_room(monkeypatch):
    # One replica runs on n0, one is Pending: a scale-up to 3 needs room for two pods, and only one fits on n1.
    errors = preflight(monkeypatch, _nodes(8, 8), _pods(("n0", "m", 8), ("", "m", 8)), rolls=False, replicas="3")
    assert errors[0].startswith("Only 1 of 2 new replicas fit")


def test_a_terminating_pod_holds_gpus_but_is_not_a_replica(monkeypatch):
    # Right after a rollout the old pod shuts down on n0 and the new one runs on n1: a scale-up to 2 needs a free node.
    pods = _pods(("n0", "m", 8), ("n1", "m", 8))
    pods["items"][0]["metadata"]["deletionTimestamp"] = "2026-01-01T00:00:00Z"
    assert preflight(monkeypatch, _nodes(8, 8), pods, rolls=False, replicas="2")[0].startswith("No node has 8 free")


@pytest.mark.parametrize("scaling_only, rolls", [(True, False), (False, True)])
def test_gpu_pods_roll_unless_only_the_scaling_changed(monkeypatch, scaling_only, rolls):
    for k, v in {"MM_PLATFORM": "kubernetes", "MM_LLMISVC_API_VERSION": "v1alpha2",
                 "MM_CONFIG": str(cli.MM_DIR / "models.yaml")}.items():
        monkeypatch.setenv(k, v)
    seen, steps = [], []
    monkeypatch.setattr(k8s, "exists", lambda *a: True)
    monkeypatch.setattr(k8s, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, "", ""))  # the diff: changed
    monkeypatch.setattr(cli, "_only_scaling_changed", lambda m, render: scaling_only)
    monkeypatch.setattr(cli, "_running_replicas", lambda m, node="": 1)
    monkeypatch.setattr(cli, "_update_credit", lambda m, annotations, cpu: 0)
    monkeypatch.setattr(cli, "preflight_gate", lambda gate, *a, running=0, amx=False: seen.append((gate.rolls, amx)))
    _quiet_deploy(monkeypatch, steps)
    cli.deploy(cli.parse_args(["deploy", "qwen3-0-6b-cuda", "--replicas", "2", "--wait"]), "qwen3-0-6b-cuda")
    assert seen == [(rolls, False)]  # and a GPU model is never held to the catalog's AMX requirement


def test_a_forced_unchanged_gpu_reapply_credits_its_running_pods(monkeypatch):
    for k, v in {"MM_PLATFORM": "kubernetes", "MM_LLMISVC_API_VERSION": "v1alpha2",
                 "MM_CONFIG": str(cli.MM_DIR / "models.yaml"), "MM_FORCE_REDEPLOY": "1"}.items():
        monkeypatch.setenv(k, v)
    seen, steps = [], []
    monkeypatch.setattr(k8s, "exists", lambda *a: True)
    monkeypatch.setattr(cli, "_only_scaling_changed", lambda m, render: True)
    monkeypatch.setattr(cli, "_running_replicas", lambda m, node="": 1)
    monkeypatch.setattr(cli, "_upgrade_credit", lambda m, annotations, cpu: 0)
    monkeypatch.setattr(cli, "preflight_gate", lambda gate, *a, running=0, amx=False: seen.append((gate.rolls, running)))
    _quiet_deploy(monkeypatch, steps)
    cli.deploy(cli.parse_args(["deploy", "qwen3-0-6b-cuda", "--wait"]), "qwen3-0-6b-cuda")
    assert seen == [(False, 1)]  # no new pods, and the running pod's CPU and memory are not counted twice


def test_node_matches_required_terms():
    node = {"metadata": {"name": "n0", "labels": {"kubernetes.io/hostname": "n0", "zone": "a"}}}
    term = lambda *e, fields=(): [{"matchExpressions": list(e), "matchFields": list(fields)}]  # noqa: E731
    assert policy._node_matches(node, term({"key": "zone", "operator": "In", "values": ["a"]}))
    assert not policy._node_matches(node, term({"key": "zone", "operator": "NotIn", "values": ["a"]}))
    assert policy._node_matches(node, term({"key": "zone", "operator": "Exists"}))
    assert not policy._node_matches(node, term({"key": "gpu", "operator": "Exists"}))
    assert policy._node_matches(node, term(fields=[{"key": "metadata.name", "operator": "In", "values": ["n0"]}]))
    assert policy._node_matches(node, term({"key": "zone", "operator": "In", "values": ["a"]}) + term())
    assert not policy._node_matches(node, [{}])  # an empty term matches no node, as in Kubernetes


@pytest.mark.parametrize("op, bound, label, matches", [
    ("Gt", "4", "8", True), ("Gt", "8", "8", False), ("Lt", "16", "8", True), ("Lt", "8", "8", False),
    ("Gt", "4", "eight", False), ("Gt", "4", None, False), ("Lt", "x", "8", False),
])
def test_node_matches_numeric_operators(op, bound, label, matches):
    node = {"metadata": {"name": "n0", "labels": {} if label is None else {"gpus": label}}}
    terms = [{"matchExpressions": [{"key": "gpus", "operator": op, "values": [bound]}]}]
    assert policy._node_matches(node, terms) is matches


def test_model_store_reads_the_pvc_deploy_names(monkeypatch):
    calls = []
    monkeypatch.setattr(k8s, "get_json", lambda *a: calls.append(a) or None)
    assert policy._model_store("llm-inference", "model-store") == ([], False)
    assert calls == [("pvc", "model-store", "-n", "llm-inference")]


def test_the_download_job_is_pinned_to_the_fitting_nodes():
    pod = yaml.safe_load("spec:" + download._node_affinity(("n0", "n1")).replace("\n      ", "\n  "))
    terms = pod["spec"]["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
    assert terms == [{"matchFields": [{"key": "metadata.name", "operator": "In", "values": [n]}]} for n in ("n0", "n1")]
    assert download._node_affinity(()) == ""
    numeric = yaml.safe_load("spec:" + download._node_affinity(("123", "true")).replace("\n      ", "\n  "))
    values = numeric["spec"]["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]
    assert [t["matchFields"][0]["values"] for t in values["nodeSelectorTerms"]] == [["123"], ["true"]]


def test_the_rendered_download_job_carries_the_pin_and_the_toleration(monkeypatch):
    monkeypatch.setenv("MM_CONFIG", str(cli.MM_DIR / "models.yaml"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    applied = []
    monkeypatch.setattr(k8s, "apply", lambda manifest, **kw: applied.append(manifest))
    monkeypatch.setattr(k8s, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""))
    monkeypatch.setattr(k8s, "out", lambda *a, **k: "")
    monkeypatch.setattr(k8s, "get_json", lambda *a: {"status": {"succeeded": 1}})
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    monkeypatch.setattr(download, "_curl_ok", lambda *a: True)
    monkeypatch.setattr(download, "hf_model_info", lambda *a, **k: {})
    catalog = Catalog.load()
    download.download_weights(config.resolve(catalog, "qwen3-0-6b-cuda", cli.Options()), catalog, [], ("n0", "123"))
    spec = next(d for d in map(yaml.safe_load, applied) if d["kind"] == "Job")["spec"]["template"]["spec"]
    terms = spec["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
    assert terms == [{"matchFields": [{"key": "metadata.name", "operator": "In", "values": [n]}]} for n in ("n0", "123")]
    assert spec["tolerations"] == [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]
    assert {"name": "NVIDIA_VISIBLE_DEVICES", "value": "void"} in spec["containers"][0]["env"]


def test_gpu_preflight_counts_only_the_pinned_node(monkeypatch):
    # --node is a required affinity: the GPUs on other nodes do not help.
    assert preflight(monkeypatch, _nodes(None, 8), node="n0")[0].startswith("Node n0 (--node)")
    assert preflight(monkeypatch, _nodes(None, 8), node="n1") == []


def test_gpu_preflight_refuses_when_not_every_replica_fits(monkeypatch):
    assert preflight(monkeypatch, _nodes(8), replicas="2")[0].startswith("Only 1 of 2 new replicas fit")
    # A scale-up only needs room for the new replica; the running one keeps its GPUs.
    assert preflight(monkeypatch, _nodes(8, 8), _pods(("n0", "m", 8)), replicas="2") == []


def test_a_cluster_without_the_gpus_stops_the_deploy(monkeypatch):
    # A CPU-only cluster: the nodes advertise no GPUs.
    for var in ("MM_SKIP_PREFLIGHT", "MM_PREFLIGHT", "MM_NRI_PREFLIGHT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(k8s, "get_json", lambda *a: _nodes(None, None) if a == ("nodes",) else _pods())
    with pytest.raises(SystemExit):
        policy.preflight_gate(policy.GpuRuntime("model-store"), Model(**GPU_MODEL, gpus="1"), 1, "8")


def test_the_request_check_counts_gpu_nodes_with_their_taint_and_no_cpu_node(monkeypatch):
    nodes = _nodes(None, 8, taints={1: [{"key": "nvidia.com/gpu", "effect": "NoSchedule"}]})
    busy = {"items": [{"spec": {"nodeName": "n1", "containers": [{"resources": {"requests": {"cpu": "60"}}}]}}]}
    m = Model(**{**GPU_MODEL, "gpus": "1", "memory": "16Gi"})
    for pods, fits in (({"items": []}, True), (busy, False)):  # n1's taint is tolerated; n0, a CPU node, never counts
        monkeypatch.setattr(k8s, "get_json", lambda *a, pods=pods: nodes if a == ("nodes",) else pods)
        assert (policy._request_errors(m, "8", False) == []) is fits


def test_the_gpus_and_the_cpu_must_fit_on_the_same_node(monkeypatch, capsys):
    """n0 has the 8 GPUs, n1 has 1: alone, each check finds room, the CPU check on n1 while n0's CPUs are taken."""
    for var in ("MM_SKIP_PREFLIGHT", "MM_PREFLIGHT", "MM_NRI_PREFLIGHT"):
        monkeypatch.delenv(var, raising=False)
    busy = {"items": [{"spec": {"nodeName": "n0", "containers": [{"resources": {"requests": {"cpu": "60"}}}]}}]}
    m = Model(**{**GPU_MODEL, "gpus": "8", "memory": "16Gi"})
    for pods in ({"items": []}, busy):
        objects = {"nodes": _nodes(8, 1), "pods": pods, "pvc": None}
        monkeypatch.setattr(k8s, "get_json", lambda *a, objects=objects: objects[a[0]])
        if pods is busy:
            with pytest.raises(SystemExit):
                policy.preflight_gate(policy.GpuRuntime("model-store"), m, 8, "8")
        else:
            policy.preflight_gate(policy.GpuRuntime("model-store"), m, 8, "8")
    assert "room for 0 where its GPUs are free too" in capsys.readouterr().err


def test_gpu_preflight_refuses_when_the_cluster_cannot_be_read(monkeypatch):
    monkeypatch.setattr(k8s, "get_json", lambda *a: None)
    assert policy.GpuRuntime("model-store").preflight(Model(**GPU_MODEL, gpus="1"), 1) == [
        "Could not list the nodes and pods to check for free nvidia.com/gpu (needs get/list on nodes and pods in "
        "every namespace)."]


def test_cli_tp_wins_over_an_arg_tp():
    assert cli._engine_cli_args(cli.Options(tp="2", arg=["--tensor-parallel-size", "8"]), 2, []) == (
        ["--tensor-parallel-size", "8", "--tensor-parallel-size", "2"], 2)


def test_catalog_tensor_parallel_is_read_on_one_gpu_and_the_last_wins():
    assert cli._engine_cli_args(cli.Options(), 1, ["--tensor-parallel-size", "2"]) == ([], 2)
    assert cli._engine_cli_args(cli.Options(), 4, ["--tensor-parallel-size", "2", "--tensor-parallel-size=8"]) == ([], 8)


def test_tp_sets_the_gpu_count_unless_gpus_is_given(monkeypatch, capsys):
    monkeypatch.setenv("MM_CONFIG", str(cli.MM_DIR / "models.yaml"))
    catalog = Catalog.load()
    assert config.resolve(catalog, "llama-3-1-70b-instruct-cuda", cli.Options(tp="4")).gpus == "4"
    assert config.resolve(catalog, "llama-3-1-70b-instruct-cuda", cli.Options(tp="4", gpus="8")).gpus == "8"
    cli.deploy(cli.Options(tp="4", gpus="8", dry_run=True), "llama-3-1-70b-instruct-cuda")
    assert "4 of 8 GPUs unused (tensor parallel size 4); pass --gpus 4." in capsys.readouterr().err


def test_tensor_parallel_larger_than_the_gpu_count_is_refused(monkeypatch, capsys):
    monkeypatch.setenv("MM_CONFIG", str(cli.MM_DIR / "models.yaml"))
    with pytest.raises(SystemExit):
        cli.deploy(cli.Options(tp="4", gpus="1", dry_run=True), "qwen3-0-6b-cuda")
    assert "Tensor parallel size 4 needs 4 GPUs, but qwen3-0-6b-cuda requests 1" in capsys.readouterr().err


def test_gpu_resolution_precedence(monkeypatch):
    monkeypatch.setenv("MM_CONFIG", str(cli.MM_DIR / "models.yaml"))
    catalog = Catalog.load()
    resolve = lambda name, **o: config.resolve(catalog, name, cli.Options(**o))  # noqa: E731
    assert resolve("llama-3-1-70b-instruct-cuda").gpus == "8"            # models.yaml gpus
    assert resolve("llama-3-1-70b-instruct-cuda", gpus="2").gpus == "2"  # --gpus wins
    assert resolve("qwen3-0-6b", gpus="2").gpus == ""                    # CPU runtime: never set
    assert resolve("", id="org/m", server="vllm-cuda", category="llm").gpus == "1"  # category default


@pytest.mark.parametrize("name, gpus", [("nemotron-3-5-lightning-30b-a3b-cuda", 1),
                                        ("nemotron-3-super-120b-a12b-cuda", 4),
                                        ("nemotron-3-ultra-550b-a55b-cuda", 8)])
def test_nemotron_entries(monkeypatch, name, gpus):
    monkeypatch.setenv("MM_CONFIG", str(cli.MM_DIR / "models.yaml"))
    m = config.resolve(Catalog.load(), name, cli.Options())
    assert m.gpu and m.gpus == str(gpus) and m.model_id.startswith("nvidia/")
    args, tp = cli._engine_cli_args(cli.Options(), gpus, renderer.engine_args(m))
    assert tp == gpus and (args == ["--tensor-parallel-size", str(gpus)] if gpus > 1 else args == [])
    assert {"--reasoning-parser", "nemotron_v3", "--tool-call-parser", "qwen3_coder"} <= set(renderer.engine_args(m))
    assert not renderer.trust_remote_code(m, args)


if __name__ == "__main__":
    legacy = Path(sys.argv[1]).resolve() / "model-manager"
    with tempfile.TemporaryDirectory() as d:
        for p in GPU_CASES:
            (GOLDEN / f"{p.id}.txt").write_text(run(legacy, p.values[0], Path(d)))
            print(f"wrote {p.id}")


def test_gpu_cpu_and_memory_are_checked_on_the_same_node(monkeypatch):
    # n0 has the GPUs but 1 CPU; n1 has the CPUs but 1 GPU: neither can run an 8-GPU, 8-CPU pod, though each
    # cluster-wide total (8 GPUs, 64 CPUs) would suffice.
    nodes = _nodes(8, 1)
    nodes["items"][0]["status"]["allocatable"]["cpu"] = "1"
    rt = policy.GpuRuntime("model-store")
    assert preflight(monkeypatch, nodes, runtime=rt, cpu="8", memory="8Gi") == []  # the GPU check alone passes
    m = Model(**{**GPU_MODEL, "gpus": "8", "cpu": "8", "memory": "8Gi"})
    errors = policy._request_errors(m, "8", False, rt, 1)
    assert errors and "room for 0 where its GPUs are free too" in errors[0]
