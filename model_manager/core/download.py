"""Weight preparation: a Job that pulls the model from HuggingFace onto the model-store PVC."""
from __future__ import annotations

import os
import re
import subprocess  # nosec B404
import time

from . import k8s
from .config import Catalog, Model, hf_model_info, pick, text
from .renderer import proxy_env, quote, render, template, trust_remote_code
from .ui import YELLOW, RESET, abort, info, ok

TIMEOUT = 7200

# Sentinel-guarded so re-deploys skip work already on the PVC.
VLLM_PREP = """set -e
export HOME=/tmp HF_HOME=/tmp/.hf XDG_CACHE_HOME=/tmp/.cache PIP_CACHE_DIR=/tmp/.pip
if [ -f @DIR@/.download_complete ]; then echo 'Weights already present, skipping download'; exit 0; fi
command -v hf >/dev/null 2>&1 || pip install -q huggingface_hub[hf_xet]
hf download '@ID@' --local-dir @DIR@@TRC@
touch @DIR@/.download_complete"""

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


def _script(m: Model, cli_args: list[str]) -> str:
    if m.runtime == "openvino":
        # --weight-format converts raw HF weights; pre-converted OpenVINO/* repos reject it, so it is opt-in.
        wf = text(pick(m.cat, "weight_format"))
        script = OVMS_PREP.replace("@WF@", f" --weight-format {wf}" if wf else "").replace(
            "@STAGE@", f"/models/.pull-{m.name}").replace("@DEVICE@", os.environ.get("MM_OVMS_TARGET_DEVICE") or "CPU")
    else:
        script = VLLM_PREP.replace("@TRC@", TRC_PREFETCH if trust_remote_code(m, cli_args) else "")
    return script.replace("@ID@", m.model_id).replace("@NAME@", m.name).replace("@DIR@", f"/models/{m.name}")


def download_weights(m: Model, catalog: Catalog, cli_args: list[str]) -> None:
    if catalog.connectivity_url and not _curl_ok(catalog.connectivity_url, catalog.https_proxy):
        abort(f"Network check failed: cannot reach {catalog.connectivity_url} (proxy: {catalog.https_proxy or 'none'}). "
              "Check network/proxy settings in models.yaml or environment.")
    token = os.environ.get("HF_TOKEN", "")
    if not token and pick(hf_model_info(m.model_id, catalog.https_proxy), "gated"):
        abort(f"Model '{m.model_id}' is gated. Set HF_TOKEN with an authorized token.")
    info(f"Downloading model weights: {m.model_id} → PVC:/{m.name}")
    if token:  # via stdin, so the token never appears in a process listing
        secret = k8s.run("create", "secret", "generic", "hf-token", "-n", m.namespace, "--from-file=token=/dev/stdin",
                         "--dry-run=client", "-o", "yaml", stdin=token)
        if secret.returncode == 0:
            k8s.run("apply", "-f", "-", stdin=secret.stdout)

    env = [("HOME", "/models"), ("HF_HOME", "/models/.hf-cache")]
    env_items = [f"            - name: {k}\n              value: {quote(v)}" for k, v in env]
    if token:
        env_items.append("            - name: HF_TOKEN\n              valueFrom:\n                secretKeyRef:\n"
                         "                  name: hf-token\n                  key: token")
    env_items += [f"            - name: {k}\n              value: {quote(v)}" for k, v in proxy_env(catalog)]
    uid = "5000" if m.runtime == "openvino" else "1000"  # the serving image's user, so it can read the files
    job = f"download-{m.name}"
    manifest = render(
        template("download-job.yaml"), MODEL_NAME=m.name, NAMESPACE=m.namespace, MODEL_ID=m.model_id,
        PVC_NAME=catalog.pvc, DOWNLOADER_IMAGE=text(pick(m.rt, "model_downloader_image")) or m.runtime_image,
        ENV_BLOCK="          env:\n" + "\n".join(env_items), JOB_UID=uid, JOB_GID=uid,
        DOWNLOAD_CMD="\n".join("              " + line for line in _script(m, cli_args).split("\n")),
        # Small requests so the job schedules; limits track the serving envelope (OVMS converts in memory).
        DL_CPU_REQUEST="500m", DL_MEM_REQUEST="2Gi", DL_CPU_LIMIT=m.cpu or "2", DL_MEM_LIMIT=m.memory or "8Gi",
        TTL_SECONDS=os.environ.get("DOWNLOAD_JOB_TTL") or "300", TIMEOUT=str(TIMEOUT))

    k8s.run("delete", "job", job, "-n", m.namespace, "--ignore-not-found")
    k8s.apply(manifest)
    info(f"Downloading model weights: {m.model_id}")
    info(f"  {YELLOW}This may take several minutes depending on model size and your internet/proxy speed.{RESET}")
    info("  Follow progress below — do not interrupt.")
    time.sleep(3)
    pod = k8s.out("get", "pods", "-n", m.namespace, "-l", f"job-name={job}", "-o", "jsonpath={.items[0].metadata.name}")
    logs = subprocess.Popen(["kubectl", "logs", "-f", pod, "-n", m.namespace],  # nosec B603 B607
                            stderr=subprocess.DEVNULL) if pod else None
    done = k8s.run("wait", "--for=condition=complete", f"job/{job}", "-n", m.namespace, f"--timeout={TIMEOUT}s")
    if logs:
        logs.terminate()
    print(done.stdout, end="")
    if done.returncode == 0:
        return ok(f"Download complete: {m.name}")
    tail = k8s.out("logs", "-l", f"job-name={job}", "-n", m.namespace, "--tail=20")
    if re.search(r"401|access denied|gated|authorization", tail, re.I):
        abort(f"Download failed: model '{m.model_id}' requires authentication. Set HF_TOKEN with an authorized token.")
    if re.search(r"ConnectError|ConnectionError|timeout", tail, re.I):
        abort("Download failed: network error reaching HuggingFace. Check proxy settings.")
    abort(f"Download job failed. Check logs: kubectl logs -l job-name={job} -n {m.namespace}")
