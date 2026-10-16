"""kubectl wrapper. Arguments are passed as argv (never through a shell)."""
from __future__ import annotations

import json
import subprocess  # nosec B404
from typing import Any

from .ui import abort, ok

SSA = ("--server-side", "--field-manager=model-manager", "--force-conflicts")


def run(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["kubectl", *args], input=stdin, text=True,  # nosec B603 B607
                          capture_output=True, check=False)


def out(*args: str) -> str:
    """Trimmed stdout, or '' when kubectl fails."""
    r = run(*args)
    return r.stdout.strip() if r.returncode == 0 else ""


def succeeds(*args: str) -> bool:
    return run(*args).returncode == 0


def get_json(*args: str) -> Any:
    try:
        return json.loads(out("get", *args, "-o", "json") or "null")
    except ValueError:
        return None


def apply(manifest: str, ssa: bool = False, quiet: bool = False) -> None:
    r = run("apply", *(SSA if ssa else ()), "-f", "-", stdin=manifest)
    if r.returncode:
        abort(r.stderr.strip() or "kubectl apply failed")
    if not quiet:
        print(r.stdout, end="")


def delete(kind: str, name: str, namespace: str, quiet: bool = False) -> None:
    r = run("delete", kind, name, "-n", namespace, "--ignore-not-found")
    if not quiet:
        print(r.stdout, end="")


def ensure_namespace(namespace: str) -> None:
    if succeeds("get", "namespace", namespace):
        return
    r = run("create", "namespace", namespace)
    if r.returncode:
        abort(r.stderr.strip())
    print(r.stdout, end="")


def ensure_storage(namespace: str, pvc: str) -> None:
    ensure_namespace(namespace)
    phase = out("get", "pvc", pvc, "-n", namespace, "-o", "jsonpath={.status.phase}")
    if phase == "Bound":
        return ok(f"Storage ready: {pvc} ({phase})")
    if not phase:
        abort(f"PVC '{pvc}' not found in namespace '{namespace}'. Provision storage with the infrastructure "
              "playbook before deploying models.")
    if phase != "Pending":
        abort(f"PVC '{pvc}' in namespace '{namespace}' is not ready (phase: {phase}). Wait for it to become "
              "Bound or check your storage provisioner.")
    # Pending is expected under WaitForFirstConsumer: the volume binds once the model pod schedules.
    sc = out("get", "pvc", pvc, "-n", namespace, "-o", "jsonpath={.spec.storageClassName}")
    mode = out("get", "storageclass", sc, "-o", "jsonpath={.volumeBindingMode}")
    if mode != "WaitForFirstConsumer":
        abort(f"PVC '{pvc}' in namespace '{namespace}' is not ready (phase: Pending, StorageClass '{sc or '?'}' "
              f"binding '{mode or 'Immediate'}'). Check your storage provisioner.")
    ok(f"Storage ready: {pvc} (Pending — binds on first consumer, StorageClass '{sc}' uses WaitForFirstConsumer)")
