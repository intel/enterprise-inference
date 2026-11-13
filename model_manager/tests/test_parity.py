"""CLI parity with the legacy bash model-manager.

Each golden file is the legacy output for one case: exit code, stdout, stderr and
every mutating kubectl call with the manifest it applied. Regenerate them with
`python tests/test_parity.py <legacy model_manager dir>`. Each case runs the new
launcher end to end against fake kubectl/curl (tests/fakebin). Intended changes
in behaviour are declared per case as `deviations`: (regex on the legacy text,
replacement).
"""
from __future__ import annotations

import json
import os
import re
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path

import pytest

MM = Path(__file__).resolve().parents[1]
GOLDEN = MM / "tests" / "golden"
FIXTURE = str(MM / "tests" / "fixtures" / "models.yaml")


def _ready(cond: str) -> str:
    return json.dumps({"status": {"conditions": [{"type": cond, "status": "True"}]}})


DOWNLOAD = [[r"get pvc model-store .*phase", 0, "Bound"],
            [r"get job download-\S+ -n \S+ -o json$", 0, json.dumps({"status": {"succeeded": 1}})],
            [r"get pods -n \S+ -l job-name=download-", 0, "dl-pod"]]
POOL = [[r"get inferencepool", 0, ""],
        [r"refs\[0\]\.name", 0, "ai-gateway"], [r"refs\[0\]\.namespace", 0, "llm-inference"]]
QWEN_READY = [[r"get LLMInferenceService qwen3-0-6b -n llm-inference -o json$", 0, _ready("MainWorkloadReady")]]
QWEN_EXISTS = [[r"get LLMInferenceService qwen3-0-6b -n llm-inference$", 0, ""]]
STATIC = [[r"get nodes -l", 0, "w1"],
          [r"configz", 0, json.dumps({"kubeletconfig": {"cpuManagerPolicy": "static",
                                                         "cpuManagerPolicyOptions": {"full-pcpus-only": "true"}}})]]
SMALL_NODES = [[r"get nodes -o json", 0, json.dumps(
    {"items": [{"metadata": {"name": "w1"}, "spec": {}, "status": {"allocatable": {"cpu": "4"}}}]})]]
LIVE_AUTO_CPU = [[r"(?i)^get llminferenceservice auto-cpu -n ns1 -o jsonpath=\{\.spec\.template\.containers\[0\]"
                  r"\.resources\.requests\.cpu\}$", 0, "12"]]
BE = {"MM_CPU_POLICY": "best-effort"}
FX_TEMPLATES = {"MM_CHAT_TEMPLATES": str(MM / "tests" / "fixtures" / "chat-templates")}
NRI = {"MM_CPU_POLICY": "nri-balloons"}
# Legacy computed LiteLLM's api_base even with LiteLLM disabled, warning about it.
NO_LITELLM_URL = (r"^\[WARN \]  AI Gateway service 'ai-gateway' not found.*\n", "")
# --help documents MM_PLATFORM and MM_INFERENCE_GATEWAY_SVC_NS, which are new.
HELP_SVC_NS = (r"^(  MM_INFERENCE_HOSTNAME .*\n)",
               r"\1  MM_PLATFORM           openshift | kubernetes (default: detected). On\n"
               r"                        OpenShift models attach to OpenShift AI's inference\n"
               r"                        Gateway and run with the namespace's SCC UIDs.\n"
               r"  MM_INFERENCE_GATEWAY_SVC_NS\n                        Namespace of the AI Gateway Service LiteLLM calls\n"
               r"                        (default: MM_INFERENCE_GATEWAY_NS, else envoy-gateway-system)\n")
# --help documents the list command, which is new.
HELP_LIST = [
    (r"^(  undeploy <model>\|<name>\|all .*\n)",
     r"\1  list                            Catalog models (models.yaml) with their status, and deployed models not in it\n"),
    (r"^(Deploy Options:\n)",
     r"List Options:\n  --catalog             Only the catalog (models.yaml); no cluster access needed\n"
     r"  --deployed            Only models deployed in the cluster\n\n\1"),
    (r"^(  \./model-manager undeploy llama3-8b-awq --wait .*\n)",
     r"\1  ./model-manager list                                 # catalog + deployed models and their status\n"),
]
# --id became --hfid (--id kept as a deprecated alias).
HFID = (r"or --id <hf/repo>", "or --hfid <hf/repo>")
# Autoscaling (KEDA) options are new.
HELP_SCALING = [
    (r"^  --replicas N         Replica count \(default: 1\)$",
     "  --replicas N         Fixed replica count (default: 1); turns autoscaling off\n"
     "  --min-replicas N     Autoscaling (KEDA): fewest replicas (default: 1)\n"
     "  --max-replicas N     Autoscaling: most replicas; above --min-replicas turns it on\n"
     "  --scale-target N     Autoscaling: per-replica load that triggers a scale-up\n"
     "                            (vLLM: waiting requests, default 2; OVMS: in-flight requests, default 4)"),
    (r"^(  kubectl patch llminferenceservice .*)\n",
     r"\1  # fixed count only\n  kubectl get scaledobject,hpa -n llm-inference          # autoscaled models\n"),
]
# A kubelet-static pod requests cpu CPUs; tensor parallelism splits them, it does not multiply them.
KUBELET_PER_POD = (r"allocatable CPUs \(cpu=(\d+) x tp=\d+\)", r"allocatable CPUs (cpu=\1)")
# OVMS deploys as an LLMInferenceService routed through the KServe scheduler (epp), no longer as
# an InferenceService with a direct route. The legacy OVMS deploy cases (ovms-nri, deploy-ovms-wait,
# fx-servers-sorted-default) have no counterpart there; test_units covers what they checked.
# A --wait that times out exits 1 and removes the model this deploy created (legacy exited 0 and left it).
WAIT_FAILS = [
    (r"^# rc: 0$", "# rc: 1"),
    (r"^\n  ─+\n  Model:(?:.*\n)*?    https://inference\.example\.com/v1/chat/completions\n\n",
     ("[INFO ]  Rolling back: removing qwen3-0-6b, which this deploy created.\n"
      "[INFO ]  Waiting for qwen3-0-6b pods to terminate (timeout: 180s)...\n"
      "[  OK ]  qwen3-0-6b undeployed (all pods terminated, CPUs released)\n")),
    (r"^\[WARN \]  qwen3-0-6b did not become Ready within 1s\n\[WARN \]    Check progress: .*\n",
     ("[WARN ]  qwen3-0-6b is not Ready: timed out after 1s\n"
      "[ERROR]  Deploy of qwen3-0-6b failed and was rolled back: timed out after 1s\n")),
    (r"\Z", ("$ kubectl delete LLMInferenceService qwen3-0-6b -n llm-inference --ignore-not-found\n"
             "$ kubectl delete configmap qwen3-0-6b-chat-template -n llm-inference --ignore-not-found\n"
             "$ kubectl delete aigatewayroute aigw-qwen3-0-6b -n llm-inference --ignore-not-found\n"
             "$ kubectl delete job download-qwen3-0-6b -n llm-inference --ignore-not-found\n"))]
# The pre-flight names what it checks, and also checks CPU/memory requests against the nodes the model can use.
KUBELET_REQUESTS = [
    (r"^\[WARN \]  kubelet-static pre-flight issues detected:$",
     "[WARN ]  kubelet-static pre-flight: llama3-8b-awq cannot place 1 replica(s):"),
    (r"^(\[WARN \]    - No schedulable node has 16 allocatable CPUs.*\n)",
     (r"\1[WARN ]    - 1 replica(s) requesting 16 CPU and 32Gi each: the nodes it can use have room for 0 "
      "(free: no schedulable node with AMX); the rest would stay Pending.\n"))]
OVMS_KIND = (r"^\$ kubectl delete InferenceService qwen3-5-2b-ovms ",
             "$ kubectl delete LLMInferenceService qwen3-5-2b-ovms ")
OVMS_AIGATEWAY_ROUTE = (r"^\$ kubectl delete (aiservicebackend|backend) qwen3-5-2b-ovms .*\n", "")
OVMS_AGW_ROUTE = (r"^(\$ kubectl delete httproute )direct-(qwen3-5-2b-ovms )", r"\1aigw-\2")
HELP_HFID = [
    (r"^  deploy <model>\|--id <hf/repo>   ", "  deploy <model>|--hfid <hf/repo> "),
    (r"^  --id <hf/repo>       Ad-hoc deploy any HuggingFace model$",
     "  --hfid <hf/repo>     Ad-hoc deploy any HuggingFace model (--id still works, deprecated)"),
    (r"\(with --id\)$", "(with --hfid)"),
    (r"deploy --id meta-llama", "deploy --hfid meta-llama"),
]


def nri_node(in_use: int) -> list:
    """One Ready NRI worker w1: 2 NUMA nodes, 128 logical CPUs, siblings reserved; in_use balloon CPUs taken."""
    node = {"metadata": {"name": "w1", "labels": {"nri.intel.com/max-tp": "2"}}, "spec": {},
            "status": {"conditions": [{"type": "Ready", "status": "True"}]}}
    pod = {"metadata": {"annotations": {"balloon.balloons.resource-policy.nri.io/container.main": "vllm-balloon"}},
           "spec": {"containers": [{"resources": {"requests": {"cpu": str(in_use)}}}]}, "status": {"phase": "Running"}}
    return [[r"^get nodes -o json$", 0, json.dumps({"items": [node]})],
            [r"^get pods -A --field-selector spec\.nodeName=w1 -o json$", 0, json.dumps({"items": [pod]})],
            [r"numa-nodes\}$", 0, "2"], [r"physical-cpus\}$", 0, "64"], [r"allocatable\.cpu\}$", 0, "126"],
            [r"capacity\.cpu\}$", 0, "128"], [r"reservedResources\.cpu\}$", 0, "cpuset:64-127"],
            [r"^get noderesourcetopologies\S* w1 -o json$", 0, json.dumps(_nrt(in_use))],
            [r"numa-cpus\}$", 0, '{"0": "0-31,64-95", "1": "32-63,96-127"}']]


def _nrt(in_use: int) -> dict:
    """NRI's report for nri_node: the in-use balloon CPUs split evenly over its two NUMA nodes (0-31, 32-63)."""
    half = min(32, in_use // 2)
    cpus = ",".join(f"{lo}-{lo + half - 1}" for lo in (0, 32)) if half else ""
    return {"zones": [{"name": "vllm-balloon[0]", "type": "balloon", "attributes": [{"name": "cpuset", "value": cpus}]}]}


# Room for any model, so the request check (new; legacy had none) passes quietly; case rules match first.
ROOMY = [[r"^get pods -A --field-selector status\.phase!=Succeeded,status\.phase!=Failed -o json$", 0, '{"items": []}'],
         [r"^get nodes -o json$", 0, json.dumps({"items": [{"metadata": {"name": "big", "labels": {
             "feature.node.kubernetes.io/cpu-cpuid.AMXBF16": "true"}}, "spec": {},
             "status": {"allocatable": {"cpu": "1000", "memory": "4Ti"}}}]})]]


def case(name, *args, env=None, k8s=(), catalog=None, deviations=()):
    return pytest.param({"args": args, "env": env or {}, "k8s": list(k8s) + ROOMY, "catalog": catalog,
                         "deviations": deviations}, id=name)


CASES = [
    # Manifest rendering (--dry-run).
    case("qwen-best-effort", "deploy", "qwen3-0-6b", "--dry-run", env=BE),
    case("qwen-nri-advanced", "deploy", "qwen3-0-6b", "--dry-run", "--tp", "2", "--server-version", "0.19.1",
         "--arg", "--max-model-len=4096", "--arg", "--max-num-seqs", "--arg", "64", "--env", "FOO=bar",
         "--env", "VLLM_CPU_KVCACHE_SPACE=20", "--label", "team=ml", "--label", "example.com/owner=",
         "--node", "w1", "--replicas", "2", env={**NRI, "MM_NRI_MODE": "advanced", "MM_NRI_HIDE_HT": "true"}),
    case("llama-kubelet-static", "deploy", "llama3-8b-awq", "--dry-run", "--cpu", "7",
         env={"MM_CPU_POLICY": "kubelet-static"}, k8s=STATIC),
    case("llama-kubelet-millicpu", "deploy", "llama3-8b-awq", "--dry-run", "--cpu", "7500m", "--tp", "2",
         env={"MM_CPU_POLICY": "kubelet-static"}),
    case("kubelet-static-bad-cpu", "deploy", "llama3-8b-awq", "--dry-run", "--cpu", "abc",
         env={"MM_CPU_POLICY": "kubelet-static"}),
    case("embed-direct", "deploy", "bge-base-en", "--dry-run", "--cpu-policy", "BEST-EFFORT"),
    case("agent-default-model", "deploy", "qwen2-5-coder-14b", "--dry-run", env=BE),
    case("rerank-explicit-routing", "deploy", "bge-reranker", "--dry-run", "--routing", "epp", env=BE),
    case("id-embed-detected", "deploy", "--hfid", "BAAI/bge-small-en-v1.5", "--dry-run",
         env={**BE, "FAKE_HF_JSON": '{"pipeline_tag": "feature-extraction"}'}),
    case("id-named", "deploy", "--hfid", "meta-llama/Llama-3.2-1B-Instruct", "--name", "l1", "--cpu", "4",
         "--memory", "8Gi", "--server-version", "0.19.1", "--dry-run",
         env={**BE, "FAKE_HF_JSON": '{"pipeline_tag": "text-generation"}'}),
    case("id-hf-unreachable", "deploy", "--hfid", "Org/Some_Model.v2", "--dry-run", env=BE),
    case("alias", "deploy", "qwen3-0-6b", "--as", "qwen-alt", "--server-version", "0.19.1", "--dry-run", env=BE),
    case("options-before-command", "--dry-run", "--cpu", "4", "deploy", "qwen3-0-6b", env=BE),
    # Legacy reached the fallback by crashing nri_policy.py on int("4.5"); the box and manifest are kept.
    case("nri-non-integer-cpu", "deploy", "qwen3-0-6b", "--cpu", "4.5", "--dry-run", env=NRI, deviations=[
        (r"^\[WARN \]  NRI balloons DaemonSet not found.*\nTraceback[\s\S]*?ValueError: .*\n", ""),
        (r"^<MM>/model-manager: line \d+: .*\n", "")]),
    case("nri-disabled-legacy-flag", "deploy", "qwen3-0-6b", "--dry-run", env={"MM_NRI_ENABLED": "false"}),
    case("invalid-policy-env", "deploy", "qwen3-0-6b", "--dry-run", env={"MM_CPU_POLICY": "foo"}),
    case("gateway-ref-env", "deploy", "qwen3-0-6b", "--dry-run",
         env={**BE, "MM_INFERENCE_GATEWAY": "gw2", "MM_INFERENCE_GATEWAY_NS": "gwns"}),
    case("fx-inline-template", "deploy", "tmpl-inline", "--dry-run", "--tp", "2", "--arg", "--trust-remote-code",
         env=BE, catalog=FIXTURE),
    case("fx-path-template", "deploy", "tmpl-path", "--dry-run", env=BE, catalog=FIXTURE),
    case("fx-template-file", "deploy", "tmpl-file", "--dry-run", env={**BE, **FX_TEMPLATES}, catalog=FIXTURE),
    case("fx-template-file-both", "deploy", "tmpl-file-both", "--dry-run", env={**BE, **FX_TEMPLATES},
         catalog=FIXTURE),
    case("fx-template-file-missing", "deploy", "tmpl-file-missing", "--dry-run", env=BE, catalog=FIXTURE),
    case("fx-cpu-auto-best-effort", "deploy", "auto-cpu", "--dry-run", env=BE, catalog=FIXTURE),
    case("fx-cpu-auto-nri", "deploy", "auto-cpu", "--dry-run", "--tp", "2", env=NRI, catalog=FIXTURE,
         k8s=nri_node(8)),
    case("fx-cpu-auto-nri-full", "deploy", "auto-cpu", "--dry-run", env=NRI, catalog=FIXTURE, k8s=nri_node(124)),
    case("fx-cpu-auto-keep-live", "deploy", "auto-cpu", "--dry-run", env=NRI, catalog=FIXTURE,
         k8s=LIVE_AUTO_CPU + nri_node(8)),
    case("fx-cpu-auto-override", "deploy", "auto-cpu", "--dry-run", "--cpu", "10", env=NRI, catalog=FIXTURE,
         k8s=nri_node(8)),
    case("fx-version-not-allowed", "deploy", "multi", "--server", "vllm", "--server-version", "0.30.0",
         "--dry-run", env=BE, catalog=FIXTURE),
    case("fx-category-detected", "deploy", "nocat", "--dry-run",
         env={**BE, "FAKE_HF_JSON": '{"pipeline_tag": "image-text-to-text"}'}, catalog=FIXTURE),
    case("fx-version-delta", "deploy", "nocat", "--server-version", "0.30.0", "--category", "llm", "--dry-run",
         env=BE, catalog=FIXTURE),
    case("fx-bad-env", "deploy", "badenv", "--dry-run", env=BE, catalog=FIXTURE),
    case("fx-bad-template", "deploy", "badtpl", "--dry-run", env=BE, catalog=FIXTURE),
    # CLI surface and validation errors.
    case("no-args", deviations=[HELP_SVC_NS, *HELP_LIST, *HELP_HFID, *HELP_SCALING]),
    case("help", "--help", deviations=[HELP_SVC_NS, *HELP_LIST, *HELP_HFID, *HELP_SCALING]),
    case("help-command", "help", deviations=[HELP_SVC_NS, *HELP_LIST, *HELP_HFID, *HELP_SCALING]),
    case("version", "version"),
    case("version-flag", "-v"),
    case("unknown-option", "deploy", "qwen3-0-6b", "--bogus"),
    case("unknown-command", "frobnicate"),
    case("deploy-no-model", "deploy", deviations=[HFID]),
    case("undeploy-no-model", "undeploy"),
    case("bad-tp", "deploy", "qwen3-0-6b", "--tp", "x"),
    case("bad-label-key", "deploy", "qwen3-0-6b", "--label", "Bad Key=v"),
    case("label-without-value", "deploy", "qwen3-0-6b", "--label", "foo"),
    case("not-in-catalog", "deploy", "nope", env=BE),
    case("server-not-declared", "deploy", "qwen3-0-6b", "--server", "openvino", env=BE),
    case("version-not-declared", "deploy", "qwen3-0-6b", "--server-version", "9.9", env=BE),
    case("runtime-version-missing", "deploy", "llama3-8b-awq", "--server-version", "9.9", env=BE),
    case("flat-runtime-version", "deploy", "qwen3-5-2b-ovms", "--server-version", "1", env=BE),
    case("invalid-cli-policy", "deploy", "qwen3-0-6b", "--cpu-policy", "bogus", "--dry-run"),
    # Side effects (fake cluster).
    case("deploy-detached", "deploy", "qwen3-0-6b", env=BE, k8s=DOWNLOAD + POOL + QWEN_READY,
         deviations=[NO_LITELLM_URL]),
    case("deploy-not-ready-reconciler", "deploy", "qwen3-0-6b", "--wait-timeout", "1", env=BE, k8s=DOWNLOAD),
    case("deploy-wait-timeout", "deploy", "qwen3-0-6b", "--wait", "--wait-timeout", "1", env=BE, k8s=DOWNLOAD,
         deviations=WAIT_FAILS),
    case("redeploy-unchanged", "deploy", "qwen3-0-6b", env={**BE, "MM_FORCE_REDEPLOY": "0"},
         k8s=QWEN_EXISTS + POOL + QWEN_READY, deviations=[NO_LITELLM_URL]),
    case("redeploy-diff-error-ready", "deploy", "qwen3-0-6b", env=BE,
         k8s=QWEN_EXISTS + [[r"^diff", 2, ""]] + POOL + QWEN_READY, deviations=[NO_LITELLM_URL]),
    case("deploy-inline-template", "deploy", "tmpl-inline", "--wait", env=BE, catalog=FIXTURE, k8s=DOWNLOAD + POOL + [
        [r"get LLMInferenceService tmpl-inline -n ns1 -o json$", 0, _ready("MainWorkloadReady")]],
         deviations=[NO_LITELLM_URL]),
    case("deploy-template-file", "deploy", "tmpl-file", "--wait", env={**BE, **FX_TEMPLATES}, catalog=FIXTURE,
         k8s=DOWNLOAD + POOL + [[r"get LLMInferenceService tmpl-file -n ns1 -o json$", 0, _ready("MainWorkloadReady")]],
         deviations=[NO_LITELLM_URL]),
    case("deploy-draft", "deploy", "specdec", "--wait", env=BE, catalog=FIXTURE, k8s=DOWNLOAD + POOL + [
        [r"get LLMInferenceService specdec -n ns1 -o json$", 0, _ready("MainWorkloadReady")]],
         deviations=[NO_LITELLM_URL]),
    case("kubelet-static-preflight-blocks", "deploy", "llama3-8b-awq", env={"MM_CPU_POLICY": "kubelet-static"},
         k8s=STATIC + SMALL_NODES, deviations=[KUBELET_PER_POD, *KUBELET_REQUESTS]),
    case("undeploy-embed", "undeploy", "bge-base-en"),
    case("undeploy-ovms", "undeploy", "qwen3-5-2b-ovms", deviations=[OVMS_KIND, OVMS_AIGATEWAY_ROUTE]),
    case("undeploy-live-alias", "undeploy", "qwen-alt", k8s=[
        [r"get LLMInferenceService qwen-alt -n llm-inference -o jsonpath", 0, "llm-inference"],
        [r"get httproute direct-qwen-alt", 0, ""]]),
    case("undeploy-missing", "undeploy", "ghost"),
    case("undeploy-all-agentgateway", "undeploy", "all", "--wait", env={"MM_AI_GATEWAY_PROVIDER": "agentgateway"},
         deviations=[OVMS_KIND, OVMS_AGW_ROUTE]),
]


def run(launcher: Path, c: dict, tmp: Path) -> str:
    log = tmp / "kubectl.log"
    log.write_text("")
    env = {
        "PATH": f"{MM / 'tests' / 'fakebin'}:{os.environ['PATH']}", "HOME": str(tmp), "TMPDIR": str(tmp),
        "LANG": "C.UTF-8", "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
        "MM_CONFIG": c["catalog"] or str(MM / "models.yaml"),
        "MM_INFERENCE_HOSTNAME": "inference.example.com", "MM_LITELLM_ENABLED": "false",
        "MM_AI_GATEWAY_PROVIDER": "envoy", "MM_RECONCILE_LOG_DIR": str(tmp),
        "FAKE_K8S": json.dumps(c["k8s"]), "FAKE_K8S_LOG": str(log), **c["env"],
    }
    p = subprocess.run([str(launcher), *c["args"]], env=env, capture_output=True, text=True,  # nosec B603
                       stdin=subprocess.DEVNULL, timeout=180, check=False)
    out = f"# rc: {p.returncode}\n--- stdout\n{p.stdout}--- stderr\n{p.stderr}--- kubectl\n{log.read_text()}"
    for path, token in ((launcher.parent, "<MM>"), (MM, "<MM>"), (tmp, "<TMP>")):
        out = out.replace(str(path), token)
    return re.sub(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", "<TS>", out)


@pytest.mark.parametrize("c", CASES)
def test_parity(c, tmp_path, request):
    expected = (GOLDEN / f"{request.node.callspec.id}.txt").read_text()
    for pattern, repl in c["deviations"]:
        expected, n = re.subn(pattern, repl, expected, flags=re.M)
        assert n, f"stale deviation: {pattern}"
    assert run(MM / "model-manager", c, tmp_path) == expected


if __name__ == "__main__":
    legacy = Path(sys.argv[1]).resolve() / "model-manager"
    GOLDEN.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        for p in CASES:
            (GOLDEN / f"{p.id}.txt").write_text(run(legacy, p.values[0], Path(d)))
            print(f"wrote {p.id}")
