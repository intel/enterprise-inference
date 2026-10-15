"""Cluster platform (OpenShift or Kubernetes) and how it changes the rendered manifests.

detect() runs once per invocation, before anything is rendered. MM_PLATFORM=openshift|kubernetes;
empty means detected (only OpenShift serves the config.openshift.io ClusterVersion). On OpenShift
the models run on OpenShift AI's KServe, so the defaults change:
  - LLMInferenceServices attach to OpenShift AI's inference Gateway
    (openshift-ingress/openshift-ai-inference); no AI Gateway routes.
  - Pods do not pin runAsUser/runAsGroup/fsGroup: the restricted-v2 SCC assigns the
    namespace's UID range.
  - The Gateway controller reports InferencePool status itself.
Independently, LLMInferenceServices are rendered at the newest version the cluster serves
(OpenShift AI 3.3 serves only v1alpha1).
"""
from __future__ import annotations

import os
import re
import shlex

import yaml

from . import k8s

LLMISVC_V1ALPHA2 = "serving.kserve.io/v1alpha2"
_PINNED_ID = re.compile(r"^[ \t]*(runAsUser|runAsGroup|fsGroup):[ \t]*[0-9]+[ \t]*\n", re.MULTILINE)


def detect() -> None:
    """Resolve MM_PLATFORM and MM_LLMISVC_API_VERSION into the environment (inherited by the reconciler)."""
    if not os.environ.get("MM_PLATFORM"):
        resources = k8s.out("api-resources", "--api-group=config.openshift.io", "-o", "name").split()
        os.environ["MM_PLATFORM"] = ("openshift" if any(r.startswith("clusterversions") for r in resources)
                                     else "kubernetes")
    if not os.environ.get("MM_LLMISVC_API_VERSION"):
        served = k8s.out("api-versions").split()
        old = "serving.kserve.io/v1alpha1" in served and LLMISVC_V1ALPHA2 not in served
        os.environ["MM_LLMISVC_API_VERSION"] = "v1alpha1" if old else "v1alpha2"
    if is_openshift():
        os.environ.setdefault("MM_INFERENCE_GATEWAY", "openshift-ai-inference")
        os.environ.setdefault("MM_INFERENCE_GATEWAY_NS", "openshift-ingress")
        os.environ.setdefault("MM_AIGATEWAY_ENABLED", "false")


def is_openshift() -> bool:
    return os.environ.get("MM_PLATFORM") == "openshift"


def adapt(manifest: str) -> str:
    """Fit any rendered template to the platform: on OpenShift, drop pinned numeric UIDs/GIDs."""
    return _PINNED_ID.sub("", manifest) if is_openshift() else manifest


def adapt_llmisvc(manifest: str) -> str:
    """Fit a rendered LLMInferenceService to the API version the cluster serves and to OpenShift AI."""
    version = os.environ.get("MM_LLMISVC_API_VERSION", "v1alpha2")
    if version == "v1alpha2" and not is_openshift():
        return manifest
    doc = yaml.safe_load(manifest)
    if version != "v1alpha2" and doc.get("apiVersion") == LLMISVC_V1ALPHA2:
        doc["apiVersion"] = f"serving.kserve.io/{version}"
        # v1alpha1 has no spec.annotations (the pod annotations the CPU policy sets), so they move to
        # the object's own annotations. They do not reach the pods there; on OpenShift the NRI
        # balloons policy selects these pods by their kserve.io/component label instead.
        annotations = doc.get("spec", {}).pop("annotations", None)
        if annotations:
            doc["metadata"]["annotations"] = {**doc["metadata"].get("annotations", {}), **annotations}
    if is_openshift():
        _openshift_ai_template(doc)
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=1 << 16)


def _openshift_ai_template(doc: dict) -> None:
    """OpenShift AI's base template (LLMInferenceServiceConfig *-kserve-config-llm-template) starts
    vLLM itself through `/bin/bash -c`:
        eval "vllm serve /mnt/models --served-model-name <spec.model.name> --port 8000
              ${VLLM_ADDITIONAL_ARGS} --ssl-certfile ... --ssl-keyfile ..."
    so container args would replace that script. Instead the flags go, shell-quoted for that eval,
    into VLLM_ADDITIONAL_ARGS; the served name becomes spec.model.name; and the probes speak HTTPS,
    because vLLM serves TLS there."""
    main = next(c for c in doc["spec"]["template"]["containers"] if c.get("name") == "main")
    served, rest, args = "", [], iter(main.pop("args", None) or [])
    for arg in args:
        if arg == "--served-model-name":
            served = next(args, "")
        elif arg.startswith("--served-model-name="):
            served = arg.split("=", 1)[1]
        else:
            rest.append(arg)
    doc["spec"].setdefault("model", {})["name"] = served or doc["metadata"]["name"]
    main["env"] = [e for e in main.get("env") or [] if e.get("name") != "VLLM_ADDITIONAL_ARGS"] + [
        {"name": "VLLM_ADDITIONAL_ARGS", "value": " ".join(shlex.quote(a) for a in rest)}]
    for probe in ("startupProbe", "livenessProbe", "readinessProbe"):
        if "httpGet" in main.get(probe, {}):
            main[probe]["httpGet"]["scheme"] = "HTTPS"
