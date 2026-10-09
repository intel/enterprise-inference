import json
import os
import subprocess
from typing import Any

from mm.ui import abort, err

class K8sClient:
    def __init__(self, kubeconfig: str | None = None, context: str | None = None):
        self.kubeconfig = kubeconfig or os.getenv("KUBECONFIG")
        self.context = context

    def _cmd(self, *args: str) -> list[str]:
        cmd = ["kubectl"]
        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])
        if self.context:
            cmd.extend(["--context", self.context])
        cmd.extend(args)
        return cmd

    def run(self, *args: str, check: bool = False, capture_output: bool = True) -> subprocess.CompletedProcess[str]:
        cmd = self._cmd(*args)
        try:
            res = subprocess.run(cmd, text=True, capture_output=capture_output, check=check)
            return res
        except subprocess.CalledProcessError as e:
            if check:
                raise
            return subprocess.CompletedProcess(cmd, e.returncode, e.stdout or "", e.stderr or "")

    def get_raw(self, *args: str) -> str:
        res = self.run("get", *args)
        return res.stdout.strip() if res.returncode == 0 else ""

    def get_json(self, *args: str) -> Any:
        res = self.run("get", *args, "-o", "json")
        if res.returncode != 0 or not res.stdout.strip():
            return None
        try:
            return json.loads(res.stdout)
        except json.JSONDecodeError:
            return None

    def apply_ssa(self, manifest: str, field_manager: str = "model-manager") -> bool:
        cmd = self._cmd("apply", "--server-side", f"--field-manager={field_manager}", "--force-conflicts", "-f", "-")
        res = subprocess.run(cmd, input=manifest, text=True, capture_output=True)
        if res.returncode != 0:
            abort(f"Server-side apply failed:\n{res.stderr}")
        return True

    def diff_ssa(self, manifest: str, field_manager: str = "model-manager") -> int:
        cmd = self._cmd("diff", "--server-side", f"--field-manager={field_manager}", "--force-conflicts", "-f", "-")
        res = subprocess.run(cmd, input=manifest, text=True, capture_output=True)
        return res.returncode

    def apply(self, manifest: str) -> bool:
        cmd = self._cmd("apply", "-f", "-")
        res = subprocess.run(cmd, input=manifest, text=True, capture_output=True)
        if res.returncode != 0:
            abort(f"kubectl apply failed:\n{res.stderr}")
        return True

    def delete(self, kind: str, name: str, namespace: str | None = None) -> bool:
        args = ["delete", kind, name, "--ignore-not-found"]
        if namespace:
            args.extend(["-n", namespace])
        res = self.run(*args)
        return res.returncode == 0

    def ensure_namespace(self, namespace: str) -> None:
        if not self.get_raw("namespace", namespace):
            self.run("create", "namespace", namespace)

    def ensure_storage(self, namespace: str, pvc_name: str = "model-store") -> None:
        self.ensure_namespace(namespace)
        phase = self.get_raw("pvc", pvc_name, "-n", namespace, "-o", "jsonpath={.status.phase}")
        if not phase:
            abort(f"PVC '{pvc_name}' not found in namespace '{namespace}'. Provision storage with the infrastructure playbook before deploying models.")
        elif phase == "Bound":
            from mm.ui import ok
            ok(f"Storage ready: {pvc_name} ({phase})")
        elif phase == "Pending":
            sc = self.get_raw("pvc", pvc_name, "-n", namespace, "-o", "jsonpath={.spec.storageClassName}")
            binding_mode = self.get_raw("storageclass", sc, "-o", "jsonpath={.volumeBindingMode}")
            if binding_mode == "WaitForFirstConsumer":
                from mm.ui import ok
                ok(f"Storage ready: {pvc_name} (Pending — binds on first consumer, StorageClass '{sc}' uses WaitForFirstConsumer)")
            else:
                abort(f"PVC '{pvc_name}' in namespace '{namespace}' is not ready (phase: Pending, StorageClass '{sc}' binding '{binding_mode}'). Check your storage provisioner.")
        else:
            abort(f"PVC '{pvc_name}' in namespace '{namespace}' is not ready (phase: {phase}). Wait for it to become Bound or check your storage provisioner.")

    def ensure_cluster_reachable(self) -> None:
        res = self.run("get", "--raw=/readyz", "--request-timeout=5s")
        if res.returncode != 0:
            err("kubectl cannot reach the Kubernetes API server.")
            if self.kubeconfig:
                err(f"  Current KUBECONFIG: {self.kubeconfig}")
            else:
                err("  KUBECONFIG is not set and $HOME/.kube/config does not exist.")
            err("  Export a valid kubeconfig, e.g.:")
            err("    export KUBECONFIG=<repo-root>/env/local/kubeconfig.yaml")
            abort("kubectl is not connected to a cluster")
