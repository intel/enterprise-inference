"""Weight preparation: a Job that pulls the model from HuggingFace onto the model-store PVC."""
from __future__ import annotations

import os
import re
import subprocess  # nosec B404
import time

from . import k8s
from .stuck import Watch
from .config import HF_EXCLUDE, Catalog, Model, hf_download_bytes, hf_model_info, pick, text
from .renderer import gpu_toleration, proxy_env, quote, render, template, trust_remote_code
from .ui import YELLOW, RESET, abort, info, ok

TIMEOUT = 7200

# Sentinel-guarded so re-deploys skip work already on the PVC; the space check runs before the draft fetch.
VLLM_PREP = """set -e
export HOME=/tmp HF_HOME=/tmp/.hf XDG_CACHE_HOME=/tmp/.cache PIP_CACHE_DIR=/tmp/.pip@SPACE@@DRAFT@
if [ -f @DIR@/.download_complete ]; then echo 'Weights already present, skipping download'; exit 0; fi
command -v hf >/dev/null 2>&1 || pip install -q huggingface_hub[hf_xet]
hf download '@ID@' --local-dir @DIR@ --exclude '@EXCLUDE@'@TRC@
touch @DIR@/.download_complete"""

# Only when something will download; the excluded copy (from an older full download) is not counted as reusable.
SPACE_CHECK = """
if [ ! -f @DIR@/.download_complete ]@DRAFT_PENDING@; then
  need_kb=@NEED_KB@
  have_kb=$(du -sk --exclude=original @DIR@ 2>/dev/null | cut -f1); have_kb=${have_kb:-0}
  free_kb=$(df -Pk /models | tail -1 | { read -r _ _ _ f _; echo "$f"; })
  if [ $((need_kb - have_kb)) -gt "$free_kb" ]; then
    echo "ERROR: @NAME@ needs $((need_kb / 1048576)) GiB, the model store has $((free_kb / 1048576)) GiB free. Largest on it:"
    du -sk /models/* 2>/dev/null | sort -rn | head -5 | while read -r kb d; do echo "  $((kb / 1048576)) GiB  $d"; done
    echo "Delete an unused model's weights there (undeploy keeps them), or raise llm_services_model_store_size:"
    echo "an existing PVC grows only where its StorageClass allows volume expansion."
    exit 1
  fi
fi"""
DRAFT_PENDING = " || [ ! -f @DIR@/draft/.download_complete ]"

# The serving pod mounts only the model's directory and runs offline, so a speculative-decoding draft
# (draft_model_id) is fetched into <name>/draft, served as /mnt/models/draft. It has its own sentinel,
# so a model whose weights are already present still gets its draft.
DRAFT_FETCH = """
if [ ! -f @DIR@/draft/.download_complete ]; then
  command -v hf >/dev/null 2>&1 || pip install -q huggingface_hub[hf_xet]
  hf download '@DRAFT_ID@' --local-dir @DIR@/draft --exclude '@EXCLUDE@'
  touch @DIR@/draft/.download_complete
fi"""

# trust-remote-code serves offline (HF_HUB_OFFLINE=1), so the custom modeling code that config.json's
# auto_map points at ("org/repo--module.Class") is fetched now and auto_map rewritten to local modules.
TRC_PREFETCH = """
cfg=@DIR@/config.json
if [ -f "$cfg" ]; then
  echo 'trust-remote-code: prefetching custom modeling code'
  for repo in $(python3 -c "import json; c=json.load(open('$cfg')); print(chr(10).join({v.split('--')[0] for v in (c.get('auto_map') or {}).values() if '--' in v}))"); do
    echo "  fetching $repo"; hf download "$repo" --local-dir @DIR@ --include '*.py'
  done
  python3 -c "import json; p='$cfg'; c=json.load(open(p)); am=c.get('auto_map') or {}; c['auto_map']={k:(v.split('--')[-1] if '--' in v else v) for k,v in am.items()}; json.dump(c,open(p,'w'),indent=2)"
  echo '  patched auto_map for offline load'
fi"""

# `ovms --pull` downloads and converts to OpenVINO IR; the servable is moved to /models/<name> with
# the config_all.json the serving runtime loads via --config_path.
OVMS_PREP = '''set -e
export HOME=/models HF_HOME=/models/.hf-cache
if [ -f @DIR@/.download_complete ]; then echo 'Model already prepared, skipping'; exit 0; fi
rm -rf '@STAGE@'; mkdir -p '@STAGE@'
ovms --pull --source_model '@ID@' --model_repository_path '@STAGE@' --model_name '@NAME@' --task text_generation@WF@ --target_device @DEVICE@
graph_dir=$(dirname $(find '@STAGE@' -name graph.pbtxt | head -1))
if [ -z "$graph_dir" ]; then echo 'ERROR: pull produced no graph.pbtxt'; exit 1; fi
rm -rf '@DIR@'; mkdir -p '@DIR@'
mv "$graph_dir"/* '@DIR@'/
rm -rf '@STAGE@'
printf '%s\\n' '{ "mediapipe_config_list": [ { "name": "@NAME@", "base_path": "." } ], "model_config_list": [] }' > '@DIR@/config_all.json'
touch @DIR@/.download_complete
echo "Prepared OVMS servable + config_all.json at @DIR@"'''


def _curl_ok(url: str, proxy: str) -> bool:
    cmd = ["curl", "-sf", "--max-time", "10", *(["--proxy", proxy] if proxy else []), "-o", "/dev/null", url]
    return subprocess.run(cmd, capture_output=True, check=False).returncode == 0  # nosec B603


def _script(m: Model, cli_args: list[str], need_bytes: int = 0) -> str:
    if m.runtime == "openvino":
        # --weight-format converts raw HF weights; pre-converted OpenVINO/* repos reject it, so it is opt-in.
        wf = text(pick(m.cat, "weight_format"))
        script = OVMS_PREP.replace("@WF@", f" --weight-format {wf}" if wf else "").replace(
            "@STAGE@", f"/models/.pull-{m.name}").replace("@DEVICE@", os.environ.get("MM_OVMS_TARGET_DEVICE") or "CPU")
    else:
        space = SPACE_CHECK.replace("@NEED_KB@", str(-(-need_bytes // 1024))).replace(
            "@DRAFT_PENDING@", DRAFT_PENDING if m.draft_model_id else "") if need_bytes > 0 else ""
        script = VLLM_PREP.replace("@TRC@", TRC_PREFETCH if trust_remote_code(m, cli_args) else "").replace(
            "@DRAFT@", DRAFT_FETCH if m.draft_model_id else "").replace("@SPACE@", space).replace(
            "@EXCLUDE@", HF_EXCLUDE)
    return script.replace("@DRAFT_ID@", m.draft_model_id).replace("@ID@", m.model_id).replace(
        "@NAME@", m.name).replace("@DIR@", f"/models/{m.name}")


def _node_affinity(nodes: tuple) -> str:
    """Pin the Job to these nodes, so a volume that binds on first use lands where the GPU pod can mount it.
    One term per node: the API takes a single value in a node field selector, and terms are ORed."""
    if not nodes:
        return ""
    return ("\n      affinity:\n        nodeAffinity:\n          requiredDuringSchedulingIgnoredDuringExecution:\n"
            "            nodeSelectorTerms:" + "".join(
                "\n              - matchFields:\n                  - key: metadata.name\n"
                f"                    operator: In\n                    values:\n                      - {quote(n)}"
                for n in nodes))


def _redact(line: str) -> str:
    """A log line without credentials in URLs (a proxy can carry them)."""
    return re.sub(r"(://)[^/@\s]+@", r"\1***@", line)


def _wait_job(m: Model, job: str) -> str:
    """'' once the Job completes; 'Failed' when it failed; else why its pod is stuck, or that it timed out."""
    deadline, watch = time.monotonic() + TIMEOUT, Watch()
    while time.monotonic() < deadline:
        status = pick(k8s.get_json("job", job, "-n", m.namespace), "status", {})
        if status.get("succeeded"):
            return ""
        if any(c.get("type") == "Failed" and c.get("status") == "True" for c in status.get("conditions") or []):
            return "Failed"
        stuck = watch.stuck(pick(k8s.get_json("pods", "-n", m.namespace, "-l", f"job-name={job}"), "items", []))
        if stuck:
            return stuck
        time.sleep(5)
    return f"not finished after {TIMEOUT}s"


def download_weights(m: Model, catalog: Catalog, cli_args: list[str], nodes: tuple) -> None:
    if catalog.connectivity_url and not _curl_ok(catalog.connectivity_url, catalog.https_proxy):
        abort(f"Network check failed: cannot reach {catalog.connectivity_url} (proxy: {catalog.https_proxy or 'none'}). "
              "Check network/proxy settings in models.yaml or environment.")
    token = os.environ.get("HF_TOKEN", "")
    repos = {r: hf_model_info(r, catalog.https_proxy, blobs=True, token=token)
             for r in filter(None, (m.model_id, m.draft_model_id))}
    for repo, meta in repos.items():  # the draft downloads with the same token
        if not token and pick(meta, "gated"):
            abort(f"Model '{repo}' is gated. Set HF_TOKEN with an authorized token.")
    info(f"Downloading model weights: {m.model_id} → PVC:/{m.name}")
    if token:  # via stdin, so the token never appears in a process listing
        secret = k8s.run("create", "secret", "generic", "hf-token", "-n", m.namespace, "--from-file=token=/dev/stdin",
                         "--dry-run=client", "-o", "yaml", stdin=token)
        if secret.returncode:  # otherwise the Job waits the full timeout for a missing Secret
            abort(f"Cannot create the hf-token Secret: {secret.stderr.strip()}")
        k8s.apply(secret.stdout, quiet=True)

    env = [("HOME", "/models"), ("HF_HOME", "/models/.hf-cache")]
    if m.gpu:  # it requests no GPU, but the NVIDIA runtime shows a CUDA image (NVIDIA_VISIBLE_DEVICES=all) every GPU
        env.append(("NVIDIA_VISIBLE_DEVICES", "void"))
    env_items = [f"            - name: {k}\n              value: {quote(v)}" for k, v in env]
    if token:
        env_items.append("            - name: HF_TOKEN\n              valueFrom:\n                secretKeyRef:\n"
                         "                  name: hf-token\n                  key: token")
    env_items += [f"            - name: {k}\n              value: {quote(v)}" for k, v in proxy_env(catalog)]
    # Unknown (Hub unreachable) means no check, not a refusal.
    # Per destination: a draft that is the model itself still downloads twice (<name>/ and <name>/draft).
    need = 0 if m.runtime == "openvino" else sum(hf_download_bytes(repos[r]) for r in (m.model_id, m.draft_model_id) if r)
    uid = m.run_as_user
    # Tainted GPU nodes may be the only ones the Job can use (e.g. a node-local PVC), so it tolerates the taint.
    toleration = gpu_toleration(m, 6)
    job = f"download-{m.name}"
    manifest = render(
        template("download-job.yaml"), MODEL_NAME=m.name, NAMESPACE=m.namespace, MODEL_ID=m.model_id,
        PVC_NAME=catalog.pvc, DOWNLOADER_IMAGE=text(pick(m.rt, "model_downloader_image")) or m.runtime_image,
        ENV_BLOCK="          env:\n" + "\n".join(env_items), JOB_UID=uid, JOB_GID=uid,
        DOWNLOAD_CMD="\n".join("              " + line for line in _script(m, cli_args, need).split("\n")),
        # Small requests so the job schedules; limits track the serving envelope (OVMS converts in memory).
        DL_CPU_REQUEST="500m", DL_MEM_REQUEST="2Gi", DL_CPU_LIMIT=m.cpu or "2", DL_MEM_LIMIT=m.memory or "8Gi",
        TTL_SECONDS=os.environ.get("DOWNLOAD_JOB_TTL") or "300", TIMEOUT=str(TIMEOUT),
        TOLERATIONS="\n" + toleration if toleration else "", AFFINITY=_node_affinity(nodes))

    k8s.run("delete", "job", job, "-n", m.namespace, "--ignore-not-found")
    k8s.apply(manifest)
    info(f"Downloading model weights: {m.model_id}")
    info(f"  {YELLOW}This may take several minutes depending on model size and your internet/proxy speed.{RESET}")
    info("  Follow progress below — do not interrupt.")
    time.sleep(3)
    pod = k8s.out("get", "pods", "-n", m.namespace, "-l", f"job-name={job}", "-o", "jsonpath={.items[0].metadata.name}")
    logs = subprocess.Popen(["kubectl", "logs", "-f", pod, "-n", m.namespace],  # nosec B603 B607
                            stderr=subprocess.DEVNULL) if pod else None
    try:
        failed = _wait_job(m, job)
    finally:
        if logs:
            logs.terminate()
    if not failed:
        return ok(f"Download complete: {m.name}")
    tail = k8s.out("logs", "-l", f"job-name={job}", "-n", m.namespace, "--tail=20") if failed == "Failed" else ""
    k8s.run("delete", "job", job, "-n", m.namespace, "--ignore-not-found")  # nothing of a failed deploy stays
    if failed != "Failed":
        abort(f"Download of {m.model_id} cannot run: {failed}. Nothing was deployed.")
    if re.search(r"401|access denied|gated|authorization", tail, re.I):
        draft = f" or its draft '{m.draft_model_id}'" if m.draft_model_id else ""
        abort(f"Download failed: model '{m.model_id}'{draft} requires authentication. Set HF_TOKEN with an "
              "authorized token.")
    if re.search(r"ConnectError|ConnectionError|timeout", tail, re.I):
        abort("Download failed: network error reaching HuggingFace. Check proxy settings.")
    last = [_redact(line) for line in tail.splitlines()[-3:]]
    abort(f"Download of {m.model_id} failed; nothing was deployed. Its last log lines:\n  " + "\n  ".join(last or ["(none)"]))
