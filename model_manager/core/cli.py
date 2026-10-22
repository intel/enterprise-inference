"""model-manager CLI: deploy, undeploy and cpu-collisions-check."""
from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess  # nosec B404
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__, k8s, platform, renderer, routing
from .config import MM_DIR, Catalog, Model, is_uint, mm_path, pick, resolve, text, truthy
from .download import download_weights
from .policy import preflight_gate, resolve_policy
from .ui import BOLD, CYAN, DIM, GREEN, RESET, YELLOW, abort, fmt_duration, info, ok, warn

USAGE = f"""{BOLD}model-manager{RESET} — Deploy LLM inference models on Kubernetes

{BOLD}{CYAN}Commands:{RESET}
  {GREEN}deploy{RESET} <model>|--id <hf/repo>   Download weights + deploy serving pod
  {GREEN}cpu-collisions-check{RESET}            Report CPU pinning, NUMA layout & detect core collisions
  {GREEN}undeploy{RESET} <model>|<name>|all     Remove pod (keeps weights on PVC); accepts a catalog model, an --as alias, or any deployed name

{BOLD}{CYAN}Deploy Options:{RESET}
  {GREEN}--id{RESET} <hf/repo>       Ad-hoc deploy any HuggingFace model
  {GREEN}--name{RESET} <name>        Override generated name (with --id)
  {GREEN}--server{RESET} <name>      Model server/engine: vllm (default) | openvino  (alias: --runtime)
  {GREEN}--server-version{RESET} V   Server version (must be declared for the model if it uses servers:)
  {GREEN}--as{RESET} <name>          Deploy under a distinct name (run multiple server/version variants at once)
  {GREEN}--cpu{RESET} N              CPU cores
  {GREEN}--memory{RESET} NGi         Memory (e.g. 32Gi)
  {GREEN}--replicas{RESET} N         Replica count (default: 1)
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
  ./model-manager deploy --id meta-llama/Llama-3.2-1B-Instruct --cpu 8 --memory 16Gi
  ./model-manager undeploy llama3-8b-awq
  ./model-manager undeploy llama3-8b-awq --wait        # block until pods terminate + CPUs are freed

{BOLD}{CYAN}For status / scale / logs, use kubectl:{RESET}
  kubectl get llminferenceservices,inferenceservices -n llm-inference
  kubectl patch llminferenceservice <name> -n llm-inference -p '{{"spec":{{"replicas":N}}}}'
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
    wait_timeout: str = ""
    id: str = ""
    name: str = ""
    server: str = ""
    server_version: str = ""
    alias: str = ""
    cpu: str = ""
    memory: str = ""
    replicas: str = ""
    tp: str = ""
    cpu_policy: str = ""
    category: str = ""
    routing: str = ""
    node: str = ""
    env: list = field(default_factory=list)
    arg: list = field(default_factory=list)
    label: list = field(default_factory=list)
    positional: list = field(default_factory=list)


FLAGS = {"--dry-run": "dry_run", "--wait": "wait"}
VALUES = {"--wait-timeout": "wait_timeout", "--id": "id", "--name": "name", "--server": "server",
          "--runtime": "server", "--server-version": "server_version", "--as": "alias", "--cpu": "cpu",
          "--memory": "memory", "--replicas": "replicas", "--tp": "tp", "--cpu-policy": "cpu_policy",
          "--category": "category", "--routing": "routing", "--node": "node"}
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
            if a in ("--wait-timeout", "--replicas", "--tp") and not (is_uint(value) and int(value) > 0):
                abort(f"{a} requires a positive integer, got: {value}")
            if a == "--label":
                _check_label(value)
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


def workload_ready(m: Model) -> str:
    """True once the serving pods pass probes. MainWorkloadReady, not Ready: for an LLMInferenceService
    Ready also needs the InferencePool Accepted, which model-manager itself patches once this is True."""
    obj = k8s.get_json(m.kind, m.name, "-n", m.namespace)
    if obj is None:
        return "NotFound"
    status = {c.get("type"): c.get("status") for c in (obj.get("status") or {}).get("conditions") or []}
    return status.get("MainWorkloadReady") or status.get("Ready") or "Pending"


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


def _wait_ready(m: Model, timeout: int) -> bool:
    start, last_beat, last_msg = time.monotonic(), 0.0, ""
    while time.monotonic() - start < timeout:
        if workload_ready(m) == "True":
            return True
        elapsed = time.monotonic() - start
        if elapsed - last_beat >= 15:  # heartbeat, so a slow load does not look stuck
            msg = pod_progress(m)
            if msg != last_msg or elapsed % 60 < 15:
                info(f"  {DIM}[{fmt_duration(int(elapsed))}] {m.name}: {msg}{RESET}")
                last_msg = msg
            last_beat = elapsed
        time.sleep(5)
    return False


def _install_runtime(m: Model) -> None:
    """Install the runtime's ClusterServingRuntime on first use, with its image filled in."""
    runtimes = mm_path("MM_RUNTIMES", "runtimes")
    pointer = text(pick(m.rt, "serving_runtime_manifest"))
    path = runtimes / pointer if pointer else runtimes / m.runtime / "serving-runtime.yaml"
    if not pointer and not path.is_file():
        path = runtimes / m.runtime / "runtime.yaml"
    name = text(pick(m.rt, "runtime_name")) or f"{m.runtime}-runtime"
    if path.is_file() and not k8s.succeeds("get", "clusterservingruntime", name):
        info(f"Installing ClusterServingRuntime: {name}")
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


def deploy(o: Options, target: str) -> None:
    catalog = Catalog.load()
    m = resolve(catalog, target, o)
    m.check()
    info(f"Deploying {m.name} ({m.kind})")
    info(f"  Runtime: {m.runtime}{f' @ {m.version}' if m.version else ''} | Category: {m.category} | "
         f"Routing: {m.routing or 'default'}")
    info(f"  CPU: {m.cpu} | Memory: {m.memory} | Replicas: {m.replicas}")
    if m.node:
        info(f"  Node: {m.node}")
    policy = resolve_policy(o.cpu_policy, m.node)
    info(f"  CPU policy: {policy.name}")
    policy.prepare(m)
    cli_args, tp = _engine_cli_args(o)
    annotations, cpu = policy.annotations(m, tp)
    manifest = renderer.manifest(m, catalog, o, annotations, cpu, cli_args)
    if o.dry_run:
        return print(manifest, end="")

    # Unchanged spec: skip pre-flight and apply (the running pods would count against their own cores).
    action = "apply"
    if not truthy(os.environ.get("MM_FORCE_REDEPLOY", "")) and k8s.succeeds("get", m.kind, m.name, "-n", m.namespace):
        rc = k8s.run("diff", *k8s.SSA, "-f", "-", stdin=manifest).returncode
        if rc == 0:
            action = "skip"
            ok(f"{m.name} already deployed, spec unchanged — skipping apply")
        elif rc == 1:
            info(f"{m.name} spec changed — re-applying")
        elif workload_ready(m) == "True":
            action = "skip"
            warn(f"{m.name} spec diff failed (rc={rc}) but workload is Ready — skipping apply")
        else:
            warn(f"{m.name} spec diff failed (rc={rc}) — applying")
    if renderer.inline_chat_template(m):
        _apply_chat_template(m)
    if action == "apply":
        preflight_gate(policy, m, tp)
        k8s.ensure_storage(m.namespace, catalog.pvc)
        download_weights(m, catalog, cli_args)
        _install_runtime(m)
        k8s.apply(manifest, ssa=True)
        ok(f"Manifest applied: {m.kind}/{m.name} in {m.namespace}")
    routing.register(m)

    timeout = int(o.wait_timeout or 900)  # model load + warmup is slow
    if o.wait:
        info(f"Waiting for {m.name} to become Ready (timeout: {timeout}s)...")
        if _wait_ready(m, timeout):
            on_ready(m)
            return routing.print_endpoints(m)
        warn(f"{m.name} did not become Ready within {timeout}s")
        warn(f"  Check progress: kubectl logs -n {m.namespace} -l app.kubernetes.io/name={m.name} -f")
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
        if k8s.succeeds("get", "httproute", f"direct-{name}", "-n", ns):
            mode = "direct"
        elif k8s.succeeds("get", "aigatewayroute", f"aigw-{name}", "-n", ns) and \
                k8s.succeeds("get", "aiservicebackend", name, "-n", ns):
            mode = "aigateway"
        else:
            mode = "epp"
        info(f"Discovered live {kind}/{name} in {ns} (no catalog entry)")
        return Model(name=name, namespace=ns, kind=kind, routing=mode)
    return None


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
    k8s.delete(m.kind, m.name, m.namespace)
    k8s.delete("configmap", f"{m.name}-chat-template", m.namespace, quiet=True)
    routing.deregister(m)
    routing.litellm_deregister(m.name)
    if not o.wait:
        return ok(f"{m.name} undeployed")
    # The CR delete returns at once; pods keep their pinned CPUs until graceful shutdown completes.
    timeout = int(o.wait_timeout or UNDEPLOY_WAIT)
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


COMMANDS = {"deploy": deploy, "undeploy": undeploy, "cpu-collisions-check": cpu_collisions_check, "_reconcile": reconcile}


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # keep our lines ordered with kubectl/report output
    o = parse_args(sys.argv[1:])
    command, target = (o.positional + ["", ""])[:2]
    if command in ("", "help"):
        print(USAGE)
        sys.exit(0 if command else 1)
    if command == "version":
        return print(f"model-manager {__version__}")
    if not shutil.which("kubectl"):
        abort("kubectl not found. Install it: https://kubernetes.io/docs/tasks/tools/ — then export "
              "KUBECONFIG=<path-to-kubeconfig> to connect to your cluster.")
    if not shutil.which("curl"):
        abort("curl not found. Install it with your OS package manager.")
    if command not in COMMANDS:
        abort(f"Unknown command: {command} (run ./model-manager --help)")
    platform.detect()
    COMMANDS[command](o, target)
