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
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import cli, config, download, k8s, policy, renderer  # noqa: E402
from core.config import Catalog, Model  # noqa: E402
from core.renderer import gpu_vendor  # noqa: E402
from test_parity import BE, DOWNLOAD, GOLDEN, MM, NO_LITELLM_URL, NRI, POOL, _ready, case, run  # noqa: E402

GPU = {"MM_ACCEL_RESOURCE": "nvidia.com/gpu"}
# GPU pods are Burstable since the B300 run: no CPU limit, so kubelet's static CPU manager never pins them.
BURSTABLE_GPU = (r'^(          limits:)\n            cpu: "[^"]+"\n(            memory: \S+\n            nvidia\.com/gpu:)',
                 r"\1\n\2")
# The weight-download Job tolerates the GPU taint too, so it can run where only tainted GPU nodes fit.
GPU_DOWNLOAD_TOLERATION = (r"^(      restartPolicy: Never\n)",
                           r"\1      tolerations:\n        - key: nvidia.com/gpu\n          operator: Exists\n"
                           r"          effect: NoSchedule\n")
# The vendor preference follows the resource the manifest requests, so an ungated --dry-run shows it too.
UNGATED_VENDOR_PREFERENCE = (r"^(    tolerations:\n)",
                             r"    affinity:\n      nodeAffinity:\n"
                             r"        preferredDuringSchedulingIgnoredDuringExecution:\n"
                             r"          - weight: 100\n            preference:\n              matchExpressions:\n"
                             r"                - key: gpu-vendor\n                  operator: In\n"
                             r"                  values:\n                    - nvidia\n\1")
# Only an NVIDIA accelerator in the rack profile sets MM_ACCEL_RESOURCE (vllm-cuda is the only GPU runtime).
NVIDIA_ONLY = (r"declares an accelerator: on at least one node", "declares an NVIDIA accelerator: on at least one node")
# --tp without --gpus sets the GPU count too, so a TP 4 pod no longer holds 8 GPUs.
TP_SETS_GPUS = [(r"^(\[INFO \]    GPUs: )8( x nvidia\.com/gpu)$", r"\g<1>4\2"), (r'^( +nvidia\.com/gpu: )"8"$', r'\1"4"')]
LLAMA_GPU_READY = [[r"get LLMInferenceService llama-3-1-70b-instruct-cuda -n llm-inference -o json$", 0,
                    _ready("MainWorkloadReady")]]


def _nodes(*gpus, unschedulable=(), not_ready=(), taints=None):
    """`kubectl get nodes -o json`: node n<i> with that many allocatable GPUs (None: no GPUs)."""
    return {"items": [{"metadata": {"name": f"n{i}"},
                       "spec": {"unschedulable": i in unschedulable, "taints": (taints or {}).get(i, [])},
                       "status": {"allocatable": {"cpu": "64", **({} if g is None else {"nvidia.com/gpu": str(g)})},
                                  "conditions": [{"type": "Ready", "status": "False" if i in not_ready else "True"}]}}
                      for i, g in enumerate(gpus)]}


def _pods(*held):
    """`kubectl get pods -A -o json`: (node, model name, GPUs requested) per pod, in namespace n."""
    return {"items": [{"metadata": {"namespace": "n", "labels": {"app.kubernetes.io/name": name}},
                       "spec": {"nodeName": node, "containers": [{"resources": {"requests": {"nvidia.com/gpu": str(n)}}}]}}
                      for node, name, n in held]}


GPU_NODES = [[r"get nodes -o json$", 0, json.dumps(_nodes(8))],
             [r"get pods -A --field-selector \S+ -o json$", 0, json.dumps(_pods())]]

GPU_CASES = [
    # gpu-llama-70b is the manifest that served on B300.
    case("gpu-llama-70b", "deploy", "llama-3-1-70b-instruct-cuda", "--dry-run", env=GPU, deviations=[BURSTABLE_GPU]),
    case("gpu-dry-run-ungated", "deploy", "qwen3-0-6b-cuda", "--dry-run",
         deviations=[BURSTABLE_GPU, UNGATED_VENDOR_PREFERENCE]),
    case("gpu-count-node", "deploy", "qwen3-0-6b-cuda", "--gpus", "2", "--node", "w1", "--dry-run", env={**GPU, **NRI},
         deviations=[BURSTABLE_GPU]),
    case("gpu-explicit-tp", "deploy", "llama-3-1-70b-instruct-cuda", "--tp", "4", "--dry-run", env=GPU,
         deviations=[BURSTABLE_GPU, *TP_SETS_GPUS]),
    case("gpu-flag-ignored-on-cpu", "deploy", "qwen3-0-6b", "--gpus", "2", "--dry-run", env=BE),
    case("gpu-gate-refuses", "deploy", "qwen3-0-6b-cuda", deviations=[NVIDIA_ONLY]),
    case("gpu-bad-count", "deploy", "qwen3-0-6b-cuda", "--gpus", "x", env=GPU),
    case("gpu-deploy-wait", "deploy", "llama-3-1-70b-instruct-cuda", "--wait", env=GPU,
         k8s=GPU_NODES + DOWNLOAD + POOL + LLAMA_GPU_READY,
         deviations=[NO_LITELLM_URL, BURSTABLE_GPU, GPU_DOWNLOAD_TOLERATION]),
]


@pytest.mark.parametrize("c", GPU_CASES)
def test_gpu_parity(c, tmp_path, request):
    expected = (GOLDEN / f"{request.node.callspec.id}.txt").read_text()
    for pattern, repl in c["deviations"]:
        expected, n = re.subn(pattern, repl, expected, flags=re.M)
        assert n, f"stale deviation: {pattern}"
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


@pytest.mark.parametrize("cluster", ["gpu.intel.com/xe", "amd.com/gpu", "habana.ai/gaudi"])
def test_gate_refuses_another_vendor(monkeypatch, cluster):
    # vllm-cuda runs the CUDA image: on an Intel or AMD rack it must not request those devices.
    monkeypatch.setenv("MM_ACCEL_RESOURCE", cluster)
    m = Model(name="m", namespace="n", runtime="vllm-cuda", gpus="1",
              rt={"device": "gpu", "accelerator_resource": "nvidia.com/gpu"})
    with pytest.raises(SystemExit):
        cli._gpu_gate(m, dry_run=False)
    cli._gpu_gate(m, dry_run=True)  # a preview warns and renders


def test_gpu_runtime_only_as_llm_inference_service(monkeypatch):
    monkeypatch.setenv("MM_ACCEL_RESOURCE", "nvidia.com/gpu")
    m = Model(name="m", namespace="n", runtime="vllm-cuda", gpus="2", kind="InferenceService",
              rt={"device": "gpu", "accelerator_resource": "nvidia.com/gpu"})
    with pytest.raises(SystemExit):
        cli._gpu_gate(m, dry_run=True)


def test_free_gpus_are_never_reported_negative(monkeypatch):
    assert "the most on one node is 0" in preflight(monkeypatch, _nodes(4), _pods(("n0", "other", 8)))[0]


def test_gate_accepts_the_runtimes_vendor(monkeypatch):
    monkeypatch.setenv("MM_ACCEL_RESOURCE", "nvidia.com/gpu")
    cli._gpu_gate(Model(name="m", namespace="n", runtime="vllm-cuda", gpus="1",
                        rt={"device": "gpu", "accelerator_resource": "nvidia.com/gpu"}), dry_run=False)


@pytest.mark.parametrize("resource, vendor", [("nvidia.com/gpu", "nvidia"), ("nvidia.com/mig-1g.10gb", "nvidia"),
                                              ("gpu.intel.com/xe", ""), ("", ""), ("nvidia.com", "")])
def test_gpu_vendor_label(resource, vendor):
    assert gpu_vendor(resource) == vendor


def test_accel_resource_prefers_cluster(monkeypatch):
    m = Model(name="m", namespace="n", rt={"device": "gpu", "accelerator_resource": "nvidia.com/gpu"})
    monkeypatch.delenv("MM_ACCEL_RESOURCE", raising=False)
    assert m.gpu and m.accel_resource == "nvidia.com/gpu"  # --dry-run preview default
    monkeypatch.setenv("MM_ACCEL_RESOURCE", "nvidia.com/mig-1g.10gb")
    assert m.accel_resource == "nvidia.com/mig-1g.10gb"
    assert not Model(name="m", namespace="n", rt={}).gpu


@pytest.mark.parametrize("gpus", ["0", "two", ""])
def test_invalid_catalog_gpu_count_refused(monkeypatch, gpus):
    # --gpus is checked by the parser; a catalog or category value only reaches the gate.
    monkeypatch.setenv("MM_ACCEL_RESOURCE", "nvidia.com/gpu")
    m = Model(name="m", namespace="n", runtime="vllm-cuda", rt={"device": "gpu"}, gpus=gpus)
    with pytest.raises(SystemExit):
        cli._gpu_gate(m, dry_run=True)


def test_gpu_gate_is_inert_on_cpu(monkeypatch):
    monkeypatch.delenv("MM_ACCEL_RESOURCE", raising=False)
    cli._gpu_gate(Model(name="m", namespace="n", runtime="vllm", rt={"device": "cpu"}), dry_run=False)


GPU_MODEL = {"name": "m", "namespace": "n", "runtime": "vllm-cuda",
             "rt": {"device": "gpu", "accelerator_resource": "nvidia.com/gpu", "accelerator_toleration_key": "nvidia.com/gpu"}}
GPU_TAINT = {"key": "nvidia.com/gpu", "effect": "NoSchedule"}


def preflight(monkeypatch, nodes, pods=None, pvc=None, pv=None, binding="", rolls=True, runtime=None, **model):
    """GpuRuntime.preflight against a fake cluster: nodes, pods, the model-store PVC and its PV."""
    objects = {"nodes": nodes, "pods": pods or _pods(), "pvc": pvc, "pv": pv}
    monkeypatch.setattr(k8s, "get_json", lambda *a: objects[a[0]])
    monkeypatch.setattr(k8s, "out", lambda *a: binding if a[:2] == ("get", "storageclass") else "")
    runtime = runtime or policy.GpuRuntime()
    runtime.rolls = rolls
    return runtime.preflight(Model(**{**GPU_MODEL, "uri": "pvc://model-store/m", "gpus": "8", **model}), 1)


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
])
def test_gpu_preflight_and_the_model_store(monkeypatch, pvc, pv, binding, fits, download):
    nodes = _labelled({"items": _nodes(8)["items"] + _nodes(None)["items"]})
    nodes["items"][1]["metadata"] = {"name": "cpu1", "labels": {"kubernetes.io/hostname": "cpu1"}}
    runtime = policy.GpuRuntime()
    errors = preflight(monkeypatch, nodes, pvc=pvc, pv=pv, binding=binding, runtime=runtime)
    assert (errors == []) is fits and runtime.download_nodes == download
    if not fits:
        assert "GPU models need shared (ReadWriteMany) storage, or the model store on a GPU node." in errors[0]


def test_own_pods_on_the_stores_node_say_undeploy_not_storage(monkeypatch):
    nodes = _labelled(_nodes(8, 8))
    errors = preflight(monkeypatch, nodes, _pods(("n0", "m", 8)), pvc=BOUND, pv=_pv("n0"))
    assert errors[0].startswith("m's own running pods hold")


def test_a_store_that_binds_on_first_use_holds_replicas_on_one_node(monkeypatch, capsys):
    runtime = policy.GpuRuntime()
    assert preflight(monkeypatch, _labelled(_nodes(8, 8)), pvc=PENDING, binding="WaitForFirstConsumer",
                     replicas="3", runtime=runtime) == []
    err = capsys.readouterr().err
    assert "only 1 of 3 new replicas fit" in err and runtime.download_nodes == ("n0", "n1")
    assert "every model using it, CPU ones included, must then run on that node" in err


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


def test_a_pending_replica_still_needs_room(monkeypatch, capsys):
    # One replica runs on n0, one is Pending: a scale-up to 3 needs room for two pods, and only one fits on n1.
    assert preflight(monkeypatch, _nodes(8, 8), _pods(("n0", "m", 8), ("", "m", 8)), rolls=False, replicas="3") == []
    assert "only 1 of 2 new replicas fit" in capsys.readouterr().err


@pytest.mark.parametrize("live, diff_rc, rolls", [("2", 0, False), ("2", 1, True), ("", 0, True), ("1", 0, True)])
def test_rolls_pods_when_more_than_the_replicas_changes(monkeypatch, live, diff_rc, rolls):
    monkeypatch.setenv("MM_CONFIG", str(cli.MM_DIR / "models.yaml"))
    monkeypatch.setenv("MM_ACCEL_RESOURCE", "nvidia.com/gpu")
    m = config.resolve(Catalog.load(), "qwen3-0-6b-cuda", cli.Options(replicas="1"))
    monkeypatch.setattr(k8s, "out", lambda *a: live)
    monkeypatch.setattr(k8s, "run", lambda *a, **k: type("R", (), {"returncode": diff_rc})())
    assert cli._rolls_pods(m, Catalog.load(), cli.Options(), {}, "8", []) is rolls


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


def test_the_download_job_is_pinned_to_the_fitting_nodes():
    pod = yaml.safe_load("spec:" + download._node_affinity(("n0", "n1")).replace("\n      ", "\n  "))
    terms = pod["spec"]["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
    assert terms == [{"matchFields": [{"key": "metadata.name", "operator": "In", "values": ["n0", "n1"]}]}]
    assert download._node_affinity(()) == ""
    numeric = yaml.safe_load("spec:" + download._node_affinity(("123", "true")).replace("\n      ", "\n  "))
    values = numeric["spec"]["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]
    assert values["nodeSelectorTerms"][0]["matchFields"][0]["values"] == ["123", "true"]


def test_gpu_preflight_ignores_the_preferred_node(monkeypatch):
    assert preflight(monkeypatch, _nodes(None, 8), node="n0") == []


def test_gpu_preflight_warns_when_not_every_replica_fits(monkeypatch, capsys):
    assert preflight(monkeypatch, _nodes(8), replicas="2") == []
    assert "only 1 of 2 new replicas fit" in capsys.readouterr().err
    # A scale-up only needs room for the new replica; the running one keeps its GPUs.
    assert preflight(monkeypatch, _nodes(8, 8), _pods(("n0", "m", 8)), replicas="2") == []
    assert "replicas fit" not in capsys.readouterr().err


@pytest.mark.parametrize("have, want, refused", [("gpu.intel.com/xe", "amd.com/gpu", True),
                                                ("amd.com/gpu", "amd.com/gpu", False),
                                                ("nvidia.com/mig-1g.10gb", "nvidia.com/gpu", False)])
def test_gate_compares_the_resource_domain(monkeypatch, have, want, refused):
    monkeypatch.setenv("MM_ACCEL_RESOURCE", have)
    m = Model(name="m", namespace="n", runtime="x", gpus="1", rt={"device": "gpu", "accelerator_resource": want})
    if refused:
        with pytest.raises(SystemExit):
            cli._gpu_gate(m, dry_run=False)
    else:
        cli._gpu_gate(m, dry_run=False)


def test_declared_gpus_the_cluster_lacks_stop_the_deploy(monkeypatch):
    # MM_ACCEL_RESOURCE comes from the rack profile's accelerator; the nodes advertise no GPUs.
    monkeypatch.setenv("MM_ACCEL_RESOURCE", "nvidia.com/gpu")
    for var in ("MM_SKIP_PREFLIGHT", "MM_PREFLIGHT", "MM_NRI_PREFLIGHT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(k8s, "get_json", lambda *a: _nodes(None, None) if a == ("nodes",) else _pods())
    with pytest.raises(SystemExit):
        policy.preflight_gate(policy.GpuRuntime(), Model(**GPU_MODEL, gpus="1"), 1)


def test_gpu_preflight_skips_when_the_cluster_cannot_be_read(monkeypatch):
    monkeypatch.setattr(k8s, "get_json", lambda *a: None)
    assert policy.GpuRuntime().preflight(Model(**GPU_MODEL, gpus="1"), 1) == []


def test_cli_tp_wins_over_an_arg_tp():
    assert cli._engine_cli_args(cli.Options(tp="2", arg=["--tensor-parallel-size", "8"]), 2, []) == (
        ["--tensor-parallel-size", "8", "--tensor-parallel-size", "2"], 2)


def test_catalog_tensor_parallel_is_read_on_one_gpu_and_the_last_wins():
    assert cli._engine_cli_args(cli.Options(), 1, ["--tensor-parallel-size", "2"]) == ([], 2)
    assert cli._engine_cli_args(cli.Options(), 4, ["--tensor-parallel-size", "2", "--tensor-parallel-size=8"]) == ([], 8)


def test_tp_sets_the_gpu_count_unless_gpus_is_given(monkeypatch, capsys):
    monkeypatch.setenv("MM_CONFIG", str(cli.MM_DIR / "models.yaml"))
    monkeypatch.setenv("MM_ACCEL_RESOURCE", "nvidia.com/gpu")
    catalog = Catalog.load()
    assert config.resolve(catalog, "llama-3-1-70b-instruct-cuda", cli.Options(tp="4")).gpus == "4"
    assert config.resolve(catalog, "llama-3-1-70b-instruct-cuda", cli.Options(tp="4", gpus="8")).gpus == "8"
    cli.deploy(cli.Options(tp="4", gpus="8", dry_run=True), "llama-3-1-70b-instruct-cuda")
    assert "4 of 8 GPUs unused (tensor parallel size 4); pass --gpus 4." in capsys.readouterr().err


def test_manual_deploy_hint_names_the_runtimes_resource(monkeypatch, capsys):
    monkeypatch.delenv("MM_ACCEL_RESOURCE", raising=False)
    m = Model(name="m", namespace="n", runtime="vllm-cuda", gpus="1",
              rt={"device": "gpu", "accelerator_resource": "nvidia.com/mig-1g.10gb"})
    with pytest.raises(SystemExit):
        cli._gpu_gate(m, dry_run=False)
    assert "MM_ACCEL_RESOURCE=nvidia.com/mig-1g.10gb ./model-manager deploy m" in capsys.readouterr().err


def test_tensor_parallel_larger_than_the_gpu_count_is_refused(monkeypatch, capsys):
    monkeypatch.setenv("MM_CONFIG", str(cli.MM_DIR / "models.yaml"))
    monkeypatch.setenv("MM_ACCEL_RESOURCE", "nvidia.com/gpu")
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
