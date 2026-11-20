"""model-manager CLI: deploy, undeploy, list and cpu-collisions-check."""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess  # nosec B404
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__, k8s, platform, renderer, routing, sizer
from .config import (MM_DIR, Catalog, Model, Sizing, hf_config, hf_weight_bytes, hub_unreachable, is_uint, mm_path,
                     pick, resolve, resolve_autoscaling, text, truthy)
from .download import download_weights
from .policy import ANN, mem_bytes, memory_gate, preflight_gate, resolve_auto_cpu, resolve_policy, whole_cpus
from .stuck import Watch
from .ui import BOLD, CYAN, DIM, GREEN, RESET, YELLOW, abort, fmt_duration, info, ok, warn

USAGE = f"""{BOLD}model-manager{RESET} — Deploy LLM inference models on Kubernetes

{BOLD}{CYAN}Commands:{RESET}
  {GREEN}deploy{RESET} <model>|--hfid <hf/repo> Download weights + deploy serving pod
  {GREEN}cpu-collisions-check{RESET}            Report CPU pinning, NUMA layout & detect core collisions
  {GREEN}undeploy{RESET} <model>|<name>|all     Remove pod (keeps weights on PVC); accepts a catalog model, an --as alias, or any deployed name
  {GREEN}list{RESET}                            Catalog models (models.yaml) with their status, and deployed models not in it

{BOLD}{CYAN}List Options:{RESET}
  {GREEN}--catalog{RESET}             Only the catalog (models.yaml); no cluster access needed
  {GREEN}--deployed{RESET}            Only models deployed in the cluster

{BOLD}{CYAN}Deploy Options:{RESET}
  {GREEN}--hfid{RESET} <hf/repo>     Ad-hoc deploy any HuggingFace model (--id still works, deprecated)
  {GREEN}--name{RESET} <name>        Override generated name (with --hfid)
  {GREEN}--server{RESET} <name>      Model server/engine: vllm (default) | openvino  (alias: --runtime)
  {GREEN}--server-version{RESET} V   Server version (must be declared for the model if it uses servers:)
  {GREEN}--as{RESET} <name>          Deploy under a distinct name (run multiple server/version variants at once)
  {GREEN}--cpu{RESET} N              CPU cores
  {GREEN}--memory{RESET} NGi         Memory (e.g. 32Gi), or auto: sized from the model (vLLM llm/vlm on CPU
                            only; the default with --hfid for llm, falling back to the category memory)
  {GREEN}--kv-streams{RESET} N       With memory auto: full-context requests the KV cache holds (default 1)
  {GREEN}--replicas{RESET} N         Fixed replica count (default: 1); turns autoscaling off
  {GREEN}--min-replicas{RESET} N     Autoscaling (KEDA): fewest replicas (default: 1)
  {GREEN}--max-replicas{RESET} N     Autoscaling: most replicas; above --min-replicas turns it on
  {GREEN}--scale-target{RESET} N     Autoscaling: per-replica load that triggers a scale-up
                            (vLLM: waiting requests, default 2; OVMS: in-flight requests, default 4)
  {GREEN}--tp{RESET} N               Tensor-parallel size (splits across N NUMA nodes)
  {GREEN}--cpu-policy{RESET} <p>     CPU pinning: nri-balloons | kubelet-static | best-effort
                            (default: MM_CPU_POLICY, else auto-detected)
  {GREEN}--category{RESET} <cat>     Override category: llm | embed | rerank | vlm
  {GREEN}--routing{RESET} <mode>     Override routing: epp | aigateway | direct
  {GREEN}--node{RESET} <name>        Pin deployment to a specific node
  {GREEN}--env{RESET} KEY=VAL        Add/override env var (repeatable)
  {GREEN}--arg{RESET} "--flag=val"   Append a runtime arg (repeatable)
  {GREEN}--label{RESET} KEY=VAL      Add a metadata label to the serving CR (repeatable);
                            use it to mark ownership — managed-by=model-manager
                            is set for every caller and cannot distinguish them
  {GREEN}--wait{RESET}               deploy: block until model is Ready; undeploy: block until pods terminate
  {GREEN}--wait-timeout{RESET} N     Seconds to wait (default: deploy 900, undeploy 180)
  {GREEN}--dry-run{RESET}            Print manifest without applying

{BOLD}{CYAN}Examples:{RESET}
  ./model-manager deploy llama3-8b-awq --wait
  ./model-manager deploy llama3-8b-awq --server-version 0.19.1 --wait
  ./model-manager deploy --hfid meta-llama/Llama-3.2-1B-Instruct --cpu 8   # memory sized from the model
  ./model-manager undeploy llama3-8b-awq
  ./model-manager undeploy llama3-8b-awq --wait        # block until pods terminate + CPUs are freed
  ./model-manager list                                 # catalog + deployed models and their status

{BOLD}{CYAN}For status / scale / logs, use kubectl:{RESET}
  kubectl get llminferenceservices,inferenceservices -n llm-inference
  kubectl patch llminferenceservice <name> -n llm-inference -p '{{"spec":{{"replicas":N}}}}'  # fixed count only
  kubectl get scaledobject,hpa -n llm-inference          # autoscaled models
  kubectl logs -n llm-inference -l app.kubernetes.io/name=<name> -f

{BOLD}{CYAN}Environment:{RESET}
  {GREEN}HF_TOKEN{RESET}              HuggingFace token for gated models
  {GREEN}MM_CONFIG{RESET}             Path to models.yaml
  {GREEN}MM_CPU_POLICY{RESET}         Cluster CPU policy: nri-balloons (default) |
                        kubelet-static | best-effort. Set by the Ansible
                        install from kubernetes_cpu_policy; --cpu-policy wins.
  {GREEN}MM_NRI_ENABLED{RESET}        Legacy: false ⇒ best-effort (no pinning)
  {GREEN}MM_PREFLIGHT{RESET}          'warn' ⇒ report placement failures, don't block
  {GREEN}MM_SKIP_PREFLIGHT{RESET}     1 ⇒ skip the placement pre-flight entirely
  {GREEN}MM_INFERENCE_HOSTNAME{RESET} Override inference gateway hostname
  {GREEN}MM_PLATFORM{RESET}           openshift | kubernetes (default: detected). On
                        OpenShift models attach to OpenShift AI's inference
                        Gateway and run with the namespace's SCC UIDs.
  {GREEN}MM_INFERENCE_GATEWAY_SVC_NS{RESET}
                        Namespace of the AI Gateway Service LiteLLM calls
                        (default: MM_INFERENCE_GATEWAY_NS, else envoy-gateway-system)

{BOLD}{CYAN}CPU policies:{RESET}
  {GREEN}nri-balloons{RESET}    (default) NRI balloons plugin — NUMA-aware balloons sized
                  per TP shard. Requires the nri_cpu_balloons Ansible role.
  {GREEN}kubelet-static{RESET}  kubelet static CPU manager — pods are rendered as
                  Guaranteed QoS (integer cpu, requests == limits) so kubelet
                  grants exclusive whole cores. No plugin needed. Requires the
                  cluster installed with kubernetes_cpu_policy=kubelet-static.
  {GREEN}best-effort{RESET}     No pinning; the CFS scheduler places threads."""

LABEL_KEY = re.compile(r"([a-z0-9]([-a-z0-9.]*[a-z0-9])?/)?[a-zA-Z0-9]([-_.a-zA-Z0-9]*[a-zA-Z0-9])?")
LABEL_VALUE = re.compile(r"([a-zA-Z0-9]([-_.a-zA-Z0-9]*[a-zA-Z0-9])?)?")
UNDEPLOY_WAIT = 180  # pod shutdown is quick; an explicit --wait-timeout wins


@dataclass
class Options:
    """Every option is accepted by every command and may appear anywhere, as with the bash CLI."""
    dry_run: bool = False
    wait: bool = False
    catalog_only: bool = False
    deployed_only: bool = False
    wait_timeout: str = ""
    id: str = ""
    name: str = ""
    server: str = ""
    server_version: str = ""
    alias: str = ""
    cpu: str = ""
    memory: str = ""
    kv_streams: str = ""
    replicas: str = ""
    min_replicas: str = ""
    max_replicas: str = ""
    scale_target: str = ""
    tp: str = ""
    cpu_policy: str = ""
    category: str = ""
    routing: str = ""
    node: str = ""
    env: list = field(default_factory=list)
    arg: list = field(default_factory=list)
    label: list = field(default_factory=list)
    positional: list = field(default_factory=list)


FLAGS = {"--dry-run": "dry_run", "--wait": "wait", "--catalog": "catalog_only", "--deployed": "deployed_only"}
VALUES = {"--wait-timeout": "wait_timeout", "--hfid": "id", "--id": "id", "--name": "name", "--server": "server",
          "--runtime": "server", "--server-version": "server_version", "--as": "alias", "--cpu": "cpu",
          "--memory": "memory", "--kv-streams": "kv_streams", "--replicas": "replicas", "--tp": "tp",
          "--cpu-policy": "cpu_policy",
          "--category": "category", "--routing": "routing", "--node": "node",
          "--min-replicas": "min_replicas", "--max-replicas": "max_replicas", "--scale-target": "scale_target"}
LISTS = {"--env": "env", "--arg": "arg", "--label": "label"}


def _check_label(pair: str) -> None:
    """Reject a bad --label now rather than as a failed apply after the download."""
    if "=" not in pair:
        abort(f"--label requires KEY=VALUE, got: {pair}")
    key, value = pair.split("=", 1)
    if not key:
        abort(f"--label key must not be empty: {pair}")
    if not LABEL_KEY.fullmatch(key):
        abort(f"--label key is not a valid Kubernetes label key: {key}")
    if not LABEL_VALUE.fullmatch(value):
        abort(f"--label value is not a valid Kubernetes label value: {value}")


def parse_args(argv: list[str]) -> Options:
    o, i = Options(), 0
    while i < len(argv):
        a = argv[i]
        if a in ("-h", "--help"):
            print(USAGE)
            sys.exit(0)
        if a in ("-v", "--version"):
            print(f"model-manager {__version__}")
            sys.exit(0)
        if a == "--":
            o.positional += argv[i + 1:]
            break
        if a in FLAGS:
            setattr(o, FLAGS[a], True)
        elif a in VALUES or a in LISTS:
            if i + 1 == len(argv):
                abort(f"{a} requires a value")
            i += 1
            value = argv[i]  # taken verbatim, so `--arg --max-model-len=4096` works
            if a in ("--wait-timeout", "--replicas", "--tp", "--min-replicas", "--max-replicas",
                     "--scale-target", "--kv-streams") and not (is_uint(value) and int(value) > 0):
                abort(f"{a} requires a positive integer, got: {value}")
            if a == "--label":
                _check_label(value)
            if a == "--id":
                warn("--id is deprecated; use --hfid")
            if a in LISTS:
                getattr(o, LISTS[a]).append(value)
            else:
                setattr(o, VALUES[a], value.lower() if a == "--cpu-policy" else value)
        elif a.startswith("-"):
            abort(f"Unknown option: {a}")
        else:
            o.positional.append(a)
        i += 1
    return o


def _engine_cli_args(o: Options) -> tuple[list[str], int]:
    """--arg list (with --tp injected unless it already sets the flag) and the tensor-parallel size."""
    args = list(o.arg)
    if o.tp:
        if "--tensor-parallel-size" not in args:
            args += ["--tensor-parallel-size", o.tp]
        return args, int(o.tp)
    for n, a in enumerate(args):
        if a == "--tensor-parallel-size" or a.startswith("--tensor-parallel-size="):
            tp = a.split("=", 1)[1] if "=" in a else (args[n + 1] if n + 1 < len(args) else "1")
            if not (is_uint(tp) and int(tp) > 0):
                abort(f"--tensor-parallel-size requires a positive integer, got: {tp}")
            return args, int(tp)
    return args, 1


def _tensor_parallel(m: Model, cli_args: list[str], cli_env: list[str]) -> int:
    """TP as vLLM sees it: the last --tensor-parallel-size of its effective args (renderer.effective_args)."""
    tp = sizer.arg_value(renderer.effective_args(m, cli_args, cli_env), "--tensor-parallel-size") or "1"
    if not (is_uint(tp) and int(tp) > 0):
        abort(f"--tensor-parallel-size for {m.name} requires a positive integer, got: {tp}")
    return int(tp)


def _ready_condition(obj: dict) -> dict | None:
    """The condition that says the model serves. MainWorkloadReady, not Ready: for an LLMInferenceService
    Ready also needs the InferencePool Accepted, which model-manager itself patches once this is True.
    On OpenShift the Gateway controller reports the pool, so Ready covers the routes as well."""
    conditions = {c.get("type"): c for c in (obj.get("status") or {}).get("conditions") or []}
    if platform.is_openshift():
        return conditions.get("Ready")
    return conditions.get("MainWorkloadReady") or conditions.get("Ready")


def workload_ready(m: Model) -> str:
    """True once the model serves (see _ready_condition)."""
    obj = k8s.get_json(m.kind, m.name, "-n", m.namespace)
    if obj is None:
        return "NotFound"
    return (_ready_condition(obj) or {}).get("status") or "Pending"


def pod_progress(m: Model) -> str:
    """One-line status of the serving pod for the --wait heartbeat."""
    pods = (k8s.get_json("pods", "-n", m.namespace, "-l", f"app.kubernetes.io/name={m.name}") or {}).get("items")
    if not pods:
        return "waiting for pod to be created"
    status = pods[0].get("status") or {}
    container = (status.get("containerStatuses") or [{}])[0]
    phase, reason = status.get("phase", ""), ((container.get("state") or {}).get("waiting") or {}).get("reason", "")
    if phase == "Pending":
        return {"ContainerCreating": "creating container (pulling image / mounting weights)",
                "ImagePullBackOff": f"image pull failed ({reason})", "ErrImagePull": f"image pull failed ({reason})",
                "CrashLoopBackOff": f"container crashing ({reason}) — check logs",
                "": "pending (waiting to be scheduled)"}.get(reason, f"pending ({reason})")
    if phase == "Running":
        return "running — finalizing routes" if container.get("ready") else "running — loading model / warming up"
    return {"Failed": "pod failed — check logs", "Succeeded": "pod exited"}.get(phase, phase or "unknown")


def on_ready(m: Model) -> None:
    ok(f"{m.name} is Ready")
    if m.kind == "LLMInferenceService":
        routing.patch_pool_status(m)
    routing.litellm_register(m)


def _wait_ready(m: Model, timeout: int) -> str:
    """'' once m is Ready; else why not: a pod that cannot run, or the timeout."""
    start, last_beat, last_msg, watch = time.monotonic(), 0.0, "", Watch()
    while time.monotonic() - start < timeout:
        if workload_ready(m) == "True":
            return ""
        pods = pick(k8s.get_json("pods", "-n", m.namespace, "-l", f"app.kubernetes.io/name={m.name}"), "items", [])
        stuck = watch.stuck(pods)
        if stuck:
            return stuck
        elapsed = time.monotonic() - start
        if elapsed - last_beat >= 15:  # heartbeat, so a slow load does not look stuck
            msg = pod_progress(m)
            if msg != last_msg or elapsed % 60 < 15:
                info(f"  {DIM}[{fmt_duration(int(elapsed))}] {m.name}: {msg}{RESET}")
                last_msg = msg
            last_beat = elapsed
        time.sleep(5)
    return f"timed out after {timeout}s"


def _install_runtime(m: Model) -> None:
    """Install the runtime's ClusterServingRuntime on first use, with its image filled in."""
    runtimes = mm_path("MM_RUNTIMES", "runtimes")
    pointer = text(pick(m.rt, "serving_runtime_manifest"))
    path = runtimes / pointer if pointer else runtimes / m.runtime / "serving-runtime.yaml"
    if not pointer and not path.is_file():
        path = runtimes / m.runtime / "runtime.yaml"
    name = text(pick(m.rt, "runtime_name")) or f"{m.runtime}-runtime"
    if not path.is_file():
        return
    # Re-applied when the image changed, so an image bump reaches clusters that have the runtime.
    current = k8s.out("get", "clusterservingruntime", name, "-o", "jsonpath={.spec.containers[0].image}")
    if current == m.runtime_image:
        return
    info(f"Installing ClusterServingRuntime: {name}" if not current
         else f"Updating ClusterServingRuntime {name}: {current} -> {m.runtime_image}")
    k8s.apply(renderer.render(path, RUNTIME_IMAGE=m.runtime_image))


def _apply_chat_template(m: Model) -> None:
    """Applied on every deploy: editing the template leaves the CR spec unchanged, so a skipped apply
    would otherwise keep serving the old template."""
    k8s.ensure_namespace(m.namespace)
    info(f"Storing chat template in ConfigMap: {m.name}-chat-template")
    cm = {"apiVersion": "v1", "kind": "ConfigMap",
          "metadata": {"name": f"{m.name}-chat-template", "namespace": m.namespace,
                       "labels": {"app.kubernetes.io/managed-by": "model-manager"}},
          "data": {"chat_template.jinja": m.chat_template}}
    k8s.apply(json.dumps(cm, indent=2, ensure_ascii=False) + "\n", quiet=True)


def _spawn_reconciler(m: Model, timeout: int) -> tuple[Path, bool]:
    """Finish routing/LiteLLM registration in the background once the model turns Ready.

    The reconciler inherits an exclusive lock on its log, so a second no-wait deploy of the same
    model finds it running and starts none (two would race on LiteLLM deregister/register).
    Returns the log path and whether a reconciler was started.
    """
    log = Path(os.environ.get("MM_RECONCILE_LOG_DIR") or tempfile.gettempdir()) / f"model-manager-reconcile-{m.name}.log"
    try:  # O_NOFOLLOW: never write through a symlink planted at this predictable path
        fd = os.open(log, os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except OSError as e:
        abort(f"Cannot write {log} ({e.strerror}); re-run with --wait to finish routing for {m.name}.")
    os.fchmod(fd, 0o600)  # also for a log left by an older run
    with os.fdopen(fd, "w") as out:
        try:
            fcntl.flock(out, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return log, False
        out.truncate()
        subprocess.Popen([str(MM_DIR / "model-manager"), "_reconcile", m.name, m.namespace, m.kind,  # nosec B603
                          "--wait-timeout", str(timeout)], stdin=subprocess.DEVNULL, stdout=out, stderr=out,
                         start_new_session=True)
    return log, True


SIZED_CATEGORIES = ("llm", "vlm")


def _apply_sizing(m: Model, catalog: Catalog, o: Options, cli_args: list[str], tp: int) -> Sizing | None:
    """memory: auto sizes memory and VLLM_CPU_KVCACHE_SPACE from the model's HuggingFace metadata and aborts when it
    cannot. Otherwise the set memory and KV stay, and only their fit is checked (memory_gate) when the weight size
    is known. --hfid's implicit memory: auto (m.memory_auto_default) falls back to that instead of aborting, with a
    warning. None when nothing can be checked."""
    if not (m.vllm_on_cpu and m.category in SIZED_CATEGORIES):
        if m.memory_auto:
            abort(f"memory: auto sizes vLLM llm and vlm models on CPU only; set memory for {m.name}.")
        return None
    kv = renderer.explicit_kv_cache(m, cli_args, o.env)
    if kv and not is_uint(kv):
        abort(f"VLLM_CPU_KVCACHE_SPACE for {m.name} must be a whole number of GiB, got '{kv}'.")
    proxy, args = catalog.https_proxy, renderer.effective_args(m, cli_args, o.env)
    if m.memory_auto and kv and m.memory_auto_default:
        m.memory_auto = False  # a KV setting without --memory keeps the category memory, as for a catalog model
    if m.memory_auto and kv:
        abort(f"{m.name} has memory: auto and VLLM_CPU_KVCACHE_SPACE={kv}: memory: auto sizes both. Set memory "
              "instead, or remove the KV setting.")
    _check_kv_streams(m, args)
    inputs = _sizing_inputs(m, args, tp)
    if m.memory_auto and o.memory != "auto" and _keep_live_sizing(m, inputs):
        return None
    cfg = hf_config(m.model_id, proxy)
    if cfg and sizer.topology_error(cfg, tp):
        abort(f"{m.name}: {sizer.topology_error(cfg, tp)}; vLLM would fail at start-up.")
    weights = hf_weight_bytes(m.model_id, proxy)
    draft_weights = hf_weight_bytes(m.draft_model_id, proxy) if m.draft_model_id else 0
    s = None
    if m.memory_auto:
        draft = (hf_config(m.draft_model_id, proxy), draft_weights) if m.draft_model_id else None
        try:
            s = sizer.size(cfg, weights, args, tp, draft, int(m.kv_streams or 0))
        except (sizer.Unsupported, ValueError) as e:  # ValueError: a non-integer engine arg such as --max-num-seqs
            reason = f"Hugging Face is unreachable (proxy: {proxy or 'none'})" if hub_unreachable(proxy) else e
            if not m.memory_auto_default:
                abort(f"memory: auto cannot size {m.name}: {reason}. Set HF_TOKEN for a gated repo, or set memory "
                      "(and VLLM_CPU_KVCACHE_SPACE) for it.")
            warn(f"Memory: cannot size {m.name}: {reason}; using the {m.category} default {m.memory}. Set --memory "
                 "to choose it.")
            m.memory_auto = False
    if not s:
        draft_cfg = hf_config(m.draft_model_id, proxy) if m.draft_model_id else None
        context = sizer.served_context(cfg, args)
        load = (sizer.load_allowance_gib(cfg, weights, args, context)
                + sizer.load_allowance_gib(draft_cfg, draft_weights, args, context))
        return _static_fit(m, sizer.resident_bytes(cfg, weights, args),
                           sizer.resident_bytes(draft_cfg, draft_weights, args), int(kv or renderer.category_kv_cache(m)),
                           tp, load)
    m.memory, m.kv_cache_gb = f"{s.memory_gi}Gi", str(s.kv_gib)
    if not sizer.arg_value(args, "--max-model-len"):
        m.max_model_len = str(s.max_model_len)
    m.sizing = {SIZING_INPUTS: inputs, **({SIZED_CONTEXT: m.max_model_len} if m.max_model_len else {})}
    info(f"  Memory: {m.memory} (auto) | (weights {s.weights_gib:.1f}{_load_text(s.load_gib)} + overhead {s.tp} x "
         f"{sizer.OVERHEAD_GIB}) x {sizer.HEADROOM} + KV {s.tp} x {s.kv_gib} + {sizer.MARGIN_GIB} | KV for up to "
         f"{s.streams} x {s.max_model_len} tokens")
    return s


def _check_kv_streams(m: Model, args: list[str]) -> None:
    """kv_streams sizes the KV cache, which only memory: auto does; vLLM runs at most --max-num-seqs of them at once."""
    if not m.kv_streams:
        return
    if not m.memory_auto:
        warn(f"kv_streams {m.kv_streams} of {m.name} has no effect: it sizes the KV cache only with memory: auto.")
        return
    seqs = sizer.arg_value(args, "--max-num-seqs")
    if is_uint(seqs) and int(seqs) < int(m.kv_streams):
        warn(f"kv_streams {m.kv_streams} of {m.name} exceeds --max-num-seqs {seqs}: vLLM runs at most {seqs} requests "
             "at once, so the KV cache sized for the rest goes unused.")


SIZING_INPUTS = "memory-sizing.model-manager.io/inputs"
SIZED_CONTEXT = "memory-sizing.model-manager.io/max-model-len"


def _sizing_inputs(m: Model, args: list[str], tp: int) -> str:
    """What memory: auto sized from: the model, its draft, the image, TP, the effective engine args and kv_streams
    (only when set, so deploys sized before it existed keep their hash)."""
    key = json.dumps([m.model_id, m.draft_model_id, m.image, tp, args, *([m.kv_streams] if m.kv_streams else [])],
                     separators=(",", ":"))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _keep_live_sizing(m: Model, inputs: str) -> bool:
    """A deployed memory: auto model whose sizing inputs are unchanged keeps the memory, KV and context it was sized
    with (as cpu: auto keeps its CPU), so a re-deploy does not depend on HuggingFace or change a running model. Any
    change (TP, args, image, model) re-sizes it, as does --memory auto."""
    live = k8s.get_json(m.kind, m.name, "-n", m.namespace) if m.kind == "LLMInferenceService" else None
    annotations = pick(pick(live, "metadata", {}), "annotations", {})
    main = (pick(pick(pick(live, "spec", {}), "template", {}), "containers", []) or [{}])[0]
    memory = text(pick(pick(pick(main, "resources", {}), "limits", {}), "memory"))
    kv = next((text(e.get("value")) for e in pick(main, "env", []) if e.get("name") == renderer.KV_CACHE_ENV), "")
    if pick(annotations, SIZING_INPUTS) != inputs or not (memory and is_uint(kv)):
        return False
    m.memory, m.kv_cache_gb = memory, kv
    m.max_model_len = text(pick(annotations, SIZED_CONTEXT))
    m.sizing = {k: text(v) for k, v in annotations.items() if k in (SIZING_INPUTS, SIZED_CONTEXT)}
    info(f"  Memory: {memory} (auto, as deployed) | KV {kv} GiB per worker; --memory auto re-sizes it")
    return True


def _load_text(load_gib: float) -> str:
    return f" + load {load_gib:.1f}" if load_gib else ""


def _static_fit(m: Model, weight_bytes: int, draft_bytes: int, kv_gib: int, tp: int,
                load_gib: float = 0.0) -> Sizing | None:
    """The set memory and KV, for memory_gate to check against the weights and load allowances (the weight size is
    public even for a gated repo). None when the weight size, or a draft's, is unknown."""
    if not weight_bytes or (m.draft_model_id and not draft_bytes):
        return None
    weights = (weight_bytes + draft_bytes) / sizer.GIB
    s = Sizing(weights_gib=weights, kv_gib=kv_gib, tp=tp, max_model_len=0,
               needed_gib=weights + load_gib + tp * (kv_gib + sizer.OVERHEAD_GIB), memory_gi=0, load_gib=load_gib)
    if s.needed_gib > mem_bytes(m.memory) / sizer.GIB:
        warn(f"Memory {m.memory} of {m.name} is below the {s.needed_gib:.1f} GiB its weights {weights:.1f}"
             f"{_load_text(load_gib)} + KV {tp} x {kv_gib} + overhead {tp} x {sizer.OVERHEAD_GIB} need.")
    return s


def _require_autoscaling(m: Model) -> None:
    """KEDA installed, and a KServe whose LLMInferenceService has KEDA scaling (checked before the download)."""
    if not k8s.succeeds("get", "crd", "scaledobjects.keda.sh"):
        abort("Autoscaling needs KEDA, which is not installed: set keda_enabled: true in global_config.yaml and "
              "install the platform layer, or deploy with a fixed --replicas.")
    # kubectl explain reads the API schema through discovery, which needs no RBAC on CRDs.
    version = f"serving.kserve.io/{platform.llmisvc_api_version()}"
    if not k8s.succeeds("explain", "--api-version", version, "llminferenceservices.spec.scaling.keda"):
        abort("This cluster's KServe has no KEDA scaling for LLMInferenceService (KServe 0.21+, installed with "
              f"envoy_gateway_upgrade: true); deploy {m.name} with a fixed --replicas.")


def _running_replicas(m: Model, node: str = "") -> int:
    """Workload pods of m that hold their CPUs (on `node`, when given): on a node and Running or Pending, as NRI
    counts cores in use."""
    selector = (f"serving.kserve.io/inferenceservice={m.name}" if m.kind == "InferenceService"
                else f"app.kubernetes.io/name={m.name},kserve.io/component=workload")
    pods = pick(k8s.get_json("pods", "-n", m.namespace, "-l", selector), "items", [])
    on = [p for p in pods if pick(pick(p, "spec", {}), "nodeName") and pick(pick(p, "status", {}), "phase")
          in ("Running", "Pending")]
    return sum(1 for p in on if not node or p["spec"]["nodeName"] == node)


def _only_scaling_changed(m: Model, render) -> bool:
    """Whether m differs from the live object only in its replica count or autoscaling range."""
    spec = (k8s.get_json(m.kind, m.name, "-n", m.namespace) or {}).get("spec")
    if not spec:
        return False
    return k8s.run("diff", *k8s.SSA, "-f", "-", stdin=render(renderer.live_scaling(m, spec))).returncode == 0


def deploy(o: Options, target: str) -> None:
    catalog = Catalog.load()
    m = resolve(catalog, target, o)
    resolve_autoscaling(m, o)
    m.check()
    cli_args, _ = _engine_cli_args(o)
    tp = _tensor_parallel(m, cli_args, o.env)
    info(f"Deploying {m.name} ({m.kind})")
    info(f"  Runtime: {m.runtime}{f' @ {m.version}' if m.version else ''} | Category: {m.category} | "
         f"Routing: {m.routing or 'default'}")
    sizing = _apply_sizing(m, catalog, o, cli_args, tp)
    replicas = (f"{m.min_replicas}-{m.max_replicas} (autoscaled, target {m.scale_target} per replica)"
                if m.autoscaled else m.replicas)
    info(f"  CPU: {m.cpu} | Memory: {m.memory} | Replicas: {replicas}")
    if m.node:
        info(f"  Node: {m.node}")
    policy = resolve_policy(o.cpu_policy, m.node)
    info(f"  CPU policy: {policy.name}")
    resolve_auto_cpu(policy, m, tp)
    policy.prepare(m)
    annotations, cpu = policy.annotations(m, tp)
    manifest = renderer.manifest(m, catalog, o, annotations, cpu, cli_args)
    if o.dry_run:
        return print(manifest, end="")
    if m.autoscaled:
        _require_autoscaling(m)

    # Unchanged spec: skip pre-flight and apply (the running pods would count against their own cores).
    action, changed = "apply", False
    existed = k8s.exists(m.kind, m.name, "-n", m.namespace)  # decides the rollback: never guess on a failed get
    if existed and not truthy(os.environ.get("MM_FORCE_REDEPLOY", "")):
        rc = k8s.run("diff", *k8s.SSA, "-f", "-", stdin=manifest).returncode
        if rc == 0:
            action = "skip"
            ok(f"{m.name} already deployed, spec unchanged — skipping apply")
        elif rc == 1:
            changed = True
            info(f"{m.name} spec changed — re-applying")
        elif workload_ready(m) == "True":
            action = "skip"
            warn(f"{m.name} spec diff failed (rc={rc}) but workload is Ready — skipping apply")
        else:
            warn(f"{m.name} spec diff failed (rc={rc}) — applying")
    if renderer.inline_chat_template(m):
        _apply_chat_template(m)
    retired = False
    if action == "apply":
        def render(model: Model) -> str:
            return renderer.manifest(model, catalog, o, annotations, cpu, cli_args)

        if changed and _only_scaling_changed(m, render):
            running = _running_replicas(m)
            info("  Only the replica count or autoscaling changed: the running pods stay, so only the replicas "
                 "beyond them are checked.")
        elif changed:
            running = _update_credit(m, annotations, cpu)
            if running:
                info(f"  A rolling update replaces the running pods a few at a time: {running} of them are counted "
                     "as free, the rest are the surge pods it needs room for.")
        else:
            # An upgrade removes the old InferenceService before the new pods start, so its CPUs are free for them.
            running = _upgrade_credit(m, annotations, cpu)
            if running:
                info(f"  The previous InferenceService/{m.name} is removed before the new pods start: its {running} "
                     "running pods' CPUs and memory are counted as free.")
        preflight_gate(policy, m, tp, cpu, running=running, amx=catalog.require_amx)
        memory_gate(m, sizing)
        k8s.ensure_storage(m.namespace, catalog.pvc)
        download_weights(m, catalog, cli_args)
        if m.kind == "InferenceService":
            _install_runtime(m)
        retired = _retire_inference_service(m, wait_pods=True)
        k8s.apply(manifest, ssa=True)
        ok(f"Manifest applied: {m.kind}/{m.name} in {m.namespace}")
    routing.register(m)

    timeout = int(o.wait_timeout or 900)  # model load + warmup is slow
    if o.wait:
        info(f"Waiting for {m.name} to become Ready (timeout: {timeout}s)...")
        problem = _wait_ready(m, timeout)
        if not problem:
            on_ready(m)
            return routing.print_endpoints(m)
        _deploy_failed(m, problem, rollback=action == "apply" and not existed, retired=retired)
    else:
        time.sleep(3)
        if workload_ready(m) == "True":
            on_ready(m)
        else:
            log, started = _spawn_reconciler(m, timeout)
            info(f"{m.name} is starting in the background.")
            if not started:
                info("  A reconciler from an earlier deploy is still waiting for it; not starting another.")
            info(f"  {YELLOW}Model load + warmup can take a few minutes — try inference after a short while.{RESET}")
            info("  Registration with LiteLLM/gateway completes automatically once it is Ready."
                 if routing.litellm_enabled() else "  AI Gateway routing is reconciled automatically once it is Ready.")
            info(f"  Check status: kubectl get {m.kind} {m.name} -n {m.namespace}")
            info(f"  Follow logs:  kubectl logs -n {m.namespace} -l app.kubernetes.io/name={m.name} -f")
            info(f"  Reconcile log: {log}")
            info("  (or re-run with --wait to block until it is Ready)")
    routing.print_endpoints(m)


def _deploy_failed(m: Model, problem: str, rollback: bool, retired: bool = False) -> None:
    """Exit 1 with the reason. A model this deploy created is removed first, so nothing is left Pending."""
    warn(f"{m.name} is not Ready: {problem}")
    if rollback:
        info(f"Rolling back: removing {m.name}, which this deploy created.")
        _remove(m, wait=True, timeout=UNDEPLOY_WAIT)
        k8s.delete("job", f"download-{m.name}", m.namespace, quiet=True)  # its weights stay on the PVC
        gone = (f" The previous InferenceService/{m.name} was removed by this deploy and is not restored: deploy "
                "again once the problem is fixed.") if retired else ""
        abort(f"Deploy of {m.name} failed and was rolled back: {problem}.{gone}")
    abort(f"Deploy of {m.name} failed: {problem}. It was already deployed, so it is left as it is; check: "
          f"kubectl get pods -n {m.namespace} -l app.kubernetes.io/name={m.name}")


def reconcile(o: Options, target: str) -> None:
    """Internal: run detached by a no-wait deploy (see _spawn_reconciler)."""
    _, name, namespace, kind = (o.positional + ["", "", ""])[:4]
    m, timeout = Model(name=name, namespace=namespace, kind=kind), int(o.wait_timeout or 900)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if workload_ready(m) == "True":
            return on_ready(m)
        time.sleep(10)
    warn(f"{name} not Ready within {timeout}s — gave up before reconciling routing.")
    warn(f"  Reconcile with: ./model-manager deploy {name}")


def _discover(name: str, namespace: str) -> Model | None:
    """A live deployment with no catalog entry (e.g. an --as alias), found by name."""
    for kind in ("LLMInferenceService", "InferenceService"):
        ns = (k8s.out("get", kind, name, "-n", namespace, "-o", "jsonpath={.metadata.namespace}")
              or k8s.out("get", kind, "-A", "--field-selector", f"metadata.name={name}",
                         "-o", "jsonpath={.items[0].metadata.namespace}"))
        if not ns:
            continue
        info(f"Discovered live {kind}/{name} in {ns} (no catalog entry)")
        return Model(name=name, namespace=ns, kind=kind, routing=_live_routing(name, ns))
    return None


def _live_routing(name: str, namespace: str) -> str:
    """The routing mode a live model was deployed with, read from the routes that exist."""
    if k8s.succeeds("get", "httproute", f"direct-{name}", "-n", namespace):
        return "direct"
    if k8s.succeeds("get", "aigatewayroute", f"aigw-{name}", "-n", namespace) and \
            k8s.succeeds("get", "aiservicebackend", name, "-n", namespace):
        return "aigateway"
    return "epp"


def _owned_inference_service(m: Model) -> Model | None:
    """The InferenceService an OVMS model ran as before it moved to LLMInferenceService, when model-manager created it."""
    label = "jsonpath={.metadata.labels.app\\.kubernetes\\.io/managed-by}"
    if m.kind != "LLMInferenceService" or \
            k8s.out("get", "InferenceService", m.name, "-n", m.namespace, "-o", label) != "model-manager":
        return None
    return Model(name=m.name, namespace=m.namespace, kind="InferenceService",
                 routing=_live_routing(m.name, m.namespace))


def _fits_in_live_pods(kind: str, m: Model, annotations: dict, cpu: str) -> bool:
    """Whether a new pod of m fits where a live `kind` pod of the same name is: the same NRI balloon (which follows
    tp) and no more CPUs or memory."""
    live = k8s.get_json(kind, m.name, "-n", m.namespace) or {}
    spec = pick(live, "spec", {})
    pod = pick(spec, "predictor", {}) if kind == "InferenceService" else spec
    # v1alpha1 LLMInferenceServices carry the pod annotations on the object (platform.adapt_llmisvc).
    pod_annotations = pick(pod, "annotations", {}) or (
        pick(pick(live, "metadata", {}), "annotations", {}) if kind == "LLMInferenceService" else {})
    container = (pick(pick(pod, "model", {}), "resources", {}) if kind == "InferenceService"
                 else pick((pick(pick(spec, "template", {}), "containers", []) or [{}])[0], "resources", {}))
    requests, balloon = pick(container, "requests", {}), f"{ANN}/balloon"
    return (pick(pod_annotations, balloon) == annotations.get(balloon)
            and whole_cpus(text(pick(requests, "cpu")), floor=True) >= whole_cpus(cpu)
            and mem_bytes(text(pick(requests, "memory"))) >= mem_bytes(m.memory))


def _upgrade_credit(m: Model, annotations: dict, cpu: str) -> int:
    """Running pods of the InferenceService m replaces whose place a new replica can take. Any other change gets
    the full check."""
    old = _owned_inference_service(m)
    if old is None or not _fits_in_live_pods("InferenceService", m, annotations, cpu):
        return 0
    return _running_replicas(old, m.node)  # cores freed on another node don't help a pinned deploy


def _update_credit(m: Model, annotations: dict, cpu: str) -> int:
    """Running pods of m a rolling update replaces in place: the Deployment's 25% surge (at least one pod) needs room
    beside them, the rest is credited, when the new pods fit where the old ones are. Otherwise the full check."""
    if not _fits_in_live_pods(m.kind, m, annotations, cpu):
        return 0
    running = _running_replicas(m, m.node)
    return max(0, running - max(1, math.ceil(running / 4)))


def _retire_inference_service(m: Model, wait_pods: bool = False) -> bool:
    """Remove the InferenceService m replaces, and its routes; with `wait_pods` (a deploy), until its pods are gone
    so their CPUs are free for the new ones. Whether there was one."""
    old = _owned_inference_service(m)
    if old is None:
        return False
    info(f"Removing the previous InferenceService/{m.name}: it now deploys as an {m.kind}")
    # Delete first: if it fails, the old model keeps its routes and its cores, and the replacement is not applied.
    r = k8s.run("delete", "InferenceService", m.name, "-n", m.namespace, "--ignore-not-found")
    if r.returncode:
        abort(f"Could not remove the previous InferenceService/{m.name}: {r.stderr.strip()}")
    routing.deregister(old)
    selector = f"serving.kserve.io/inferenceservice={m.name}"
    deadline = time.monotonic() + UNDEPLOY_WAIT
    while wait_pods:
        r = k8s.run("get", "pods", "-n", m.namespace, "-l", selector, "-o", "name")
        if r.returncode:
            abort(f"Could not list the previous InferenceService/{m.name}'s pods: {r.stderr.strip()}")
        if not r.stdout.strip():
            break
        if time.monotonic() > deadline:
            abort(f"The previous InferenceService/{m.name}'s pods are still terminating after {UNDEPLOY_WAIT}s; "
                  f"re-run the deploy once they are gone: kubectl get pods -n {m.namespace} -l {selector}")
        time.sleep(3)
    return True


def undeploy(o: Options, target: str) -> None:
    if not target:
        abort("No model specified. Use: model-manager undeploy <name>|all")
    catalog = Catalog.load()
    if target == "all":
        for name in catalog.names():
            undeploy(o, name)
        return
    m = resolve(catalog, target, o) if target in catalog.names() else _discover(target, catalog.namespace)
    if m is None:
        abort(f"No catalog entry or deployed resource named '{target}' found.")
    info(f"Undeploying {m.name}")
    _retire_inference_service(m)  # first: it reads the old routes to know which to delete
    _remove(m, o.wait, int(o.wait_timeout or UNDEPLOY_WAIT))


def _remove(m: Model, wait: bool, timeout: int) -> None:
    """Delete m, its chat template and routes; with `wait`, until its pods are gone."""
    k8s.delete(m.kind, m.name, m.namespace)
    k8s.delete("configmap", f"{m.name}-chat-template", m.namespace, quiet=True)
    routing.deregister(m)
    routing.litellm_deregister(m.name)
    if not wait:
        return ok(f"{m.name} undeployed")
    # The CR delete returns at once; pods keep their pinned CPUs until graceful shutdown completes.
    info(f"Waiting for {m.name} pods to terminate (timeout: {timeout}s)...")
    start = last_beat = time.monotonic()
    while time.monotonic() - start < timeout:
        pods = k8s.out("get", "pods", "-n", m.namespace, "-l", f"app.kubernetes.io/name={m.name}", "-o", "name")
        if not pods:
            return ok(f"{m.name} undeployed (all pods terminated, CPUs released)")
        if time.monotonic() - last_beat >= 15:
            info(f"  {DIM}[{fmt_duration(int(time.monotonic() - start))}] waiting for {len(pods.splitlines())} "
                 f"pod(s) to terminate...{RESET}")
            last_beat = time.monotonic()
        time.sleep(3)
    warn(f"{m.name} undeployed, but pods are still terminating after {timeout}s — CPUs/balloons may not be reclaimed yet")
    warn(f"  Check: kubectl get pods -n {m.namespace} -l app.kubernetes.io/name={m.name}")


def _catalog_server(catalog: Catalog, e: dict) -> tuple[str, str]:
    """The server and version `deploy <name>` picks with no flags, read the way _from_catalog and
    _select_image read them, and the kind it creates."""
    servers = pick(e, "servers", {}) if isinstance(pick(e, "servers", {}), dict) else {}
    if servers:
        server = text(pick(e, "default_server")) or min(servers, default="")
        entry = pick(servers, server, {})
        declared = [text(v) for v in pick(entry, "versions", [])] if isinstance(entry, dict) else []
        version = text(pick(entry, "default", declared[0] if declared else ""))
    else:
        server, version = text(pick(e, "runtime")) or catalog.runtime, text(pick(e, "server_version"))
    rt = catalog.runtime_config(server, required=False) if server else {}
    versions = rt.get("versions")
    if not version and isinstance(versions, dict):  # the runtime's default, else its lowest version
        version = text(pick(rt, "default_version")) or min((text(k) for k in versions), default="")
    kind = text(pick(e, "kind")) or text(pick(rt, "kind", "LLMInferenceService"))
    return f"{server} {version}".strip(), kind


def _status(obj: dict) -> str:
    c = _ready_condition(obj)
    if c is None:
        return "Pending"
    return "Ready" if c.get("status") == "True" else f"Not ready ({c.get('reason')})" if c.get("reason") else "Not ready"


def _deployed(namespace: str) -> tuple[list[dict] | None, set[str]]:
    """Serving resources model-manager created, in every namespace (just the catalog's when -A is denied),
    and the kinds that could not be listed. None when neither kind can be: no cluster, no access, or
    KServe not installed."""
    found, listed, failed = [], False, set()
    for kind in ("LLMInferenceService", "InferenceService"):
        args = ("get", kind, "-l", "app.kubernetes.io/managed-by=model-manager", "-o", "json")
        r = k8s.run(*args, "-A")
        if r.returncode:
            r = k8s.run(*args, "-n", namespace)
        try:
            items = None if r.returncode else json.loads(r.stdout or "{}").get("items") or []
        except (ValueError, AttributeError):
            items = None
        if items is None:  # unknown, not absent
            failed.add(kind)
            continue
        if not listed:  # the cluster answers: which condition _status() reads depends on the platform
            platform.detect_platform()
        listed = True
        for obj in items:
            spec, meta = obj.get("spec") or {}, obj.get("metadata") or {}
            containers = ((spec.get("template") or {}).get("containers") or [{}])
            predictor = (spec.get("predictor") or {})
            found.append({"name": meta.get("name", ""), "namespace": meta.get("namespace", ""), "kind": kind,
                          "status": _status(obj),
                          "replicas": _replicas(spec, predictor),
                          "server": (containers[0].get("image") or "").rsplit("/", 1)[-1]
                          or (predictor.get("model") or {}).get("runtime", "")})
    return (found if listed else None), failed


def _replicas(spec: dict, predictor: dict) -> str:
    """Fixed count, or min-max for an autoscaled LLMInferenceService."""
    scaling = spec.get("scaling") or {}
    if scaling:
        return f"{scaling.get('minReplicas', 1)}-{scaling.get('maxReplicas', '?')}"
    return str(spec.get("replicas", predictor.get("minReplicas", "")))


def _table(header: tuple, rows: list[tuple]) -> None:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    print(BOLD + "  ".join(h.ljust(w) for h, w in zip(header, widths)).rstrip() + RESET)
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip())


def list_models(o: Options, target: str) -> None:
    if o.catalog_only and o.deployed_only:
        abort("--catalog and --deployed are exclusive; omit both to show everything.")
    catalog = Catalog.load()
    live, failed = (None, set()) if o.catalog_only else _deployed(catalog.namespace)
    if live is None and not o.catalog_only:
        warn("Cannot list deployed models (no cluster access, or KServe not installed); status shown as '?'.")
    elif failed:
        warn(f"Cannot list {', '.join(sorted(failed))}: those models show '?' and the deployed table may be incomplete.")
    by_key = {(d["name"], d["namespace"]): d for d in live or []}
    in_catalog = set()
    if not o.deployed_only:
        rows = []
        for e in catalog.models:
            name = text(pick(e, "name"))
            key = (name, text(pick(e, "namespace", catalog.namespace)))
            in_catalog.add(key)
            server, kind = _catalog_server(catalog, e)
            status = ("-" if o.catalog_only else "?" if live is None
                      else by_key[key]["status"] if key in by_key else "?" if kind in failed else "not deployed")
            model_id = text(pick(e, "model_id"))
            rows.append((f"{name} ({model_id})" if model_id else name, text(pick(e, "category")) or "auto",
                         server or "-", *(() if o.catalog_only else (status,))))
        print(f"{BOLD}{CYAN}Catalog{RESET} {DIM}({catalog.path}, {len(rows)} models){RESET}")
        _table(("MODEL ALIAS NAME (ID)", "CATEGORY", "SERVER", *(() if o.catalog_only else ("STATUS",))), rows)
    if o.catalog_only or live is None:
        return
    if not o.deployed_only:
        print()
    others = [d for d in live if o.deployed_only or (d["name"], d["namespace"]) not in in_catalog]
    title = "Deployed" if o.deployed_only else "Deployed, not in the catalog (--as / --hfid)"
    print(f"{BOLD}{CYAN}{title}{RESET} {DIM}({len(others)}){RESET}")
    if others:
        _table(("NAME", "NAMESPACE", "KIND", "SERVER", "REPLICAS", "STATUS"),
               [(d["name"], d["namespace"], d["kind"], d["server"] or "-", d["replicas"] or "-", d["status"])
                for d in sorted(others, key=lambda d: (d["namespace"], d["name"]))])
    else:
        print(f"  {DIM}none{RESET}")


def cpu_collisions_check(o: Options, target: str) -> None:
    report = MM_DIR / "scripts" / "nri-cpu-report.py"
    if not report.is_file():
        abort(f"nri-cpu-report.py not found at {report}")
    namespace = Catalog.load().namespace
    info(f"Running CPU pinning & collision report (namespace: {namespace})")
    # The kubernetes client ignores NO_PROXY and would tunnel API calls through the corporate proxy.
    env = {k: v for k, v in os.environ.items() if k.lower() not in ("http_proxy", "https_proxy")}
    sys.exit(subprocess.run([sys.executable, str(report), "--namespace", namespace],  # nosec B603
                            env=env, check=False).returncode)


COMMANDS = {"deploy": deploy, "undeploy": undeploy, "list": list_models,
            "cpu-collisions-check": cpu_collisions_check, "_reconcile": reconcile}


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # keep our lines ordered with kubectl/report output
    o = parse_args(sys.argv[1:])
    command, target = (o.positional + ["", ""])[:2]
    if command in ("", "help"):
        print(USAGE)
        sys.exit(0 if command else 1)
    if command == "version":
        return print(f"model-manager {__version__}")
    # list needs no curl, and list --catalog reads only models.yaml.
    if not shutil.which("kubectl") and not (command == "list" and o.catalog_only):
        abort("kubectl not found. Install it: https://kubernetes.io/docs/tasks/tools/ — then export "
              "KUBECONFIG=<path-to-kubeconfig> to connect to your cluster.")
    if not shutil.which("curl") and command != "list":
        abort("curl not found. Install it with your OS package manager.")
    if command not in COMMANDS:
        abort(f"Unknown command: {command} (run ./model-manager --help)")
    if command != "list":  # list renders nothing and must work without a cluster: it detects as it goes
        platform.detect()
    COMMANDS[command](o, target)
