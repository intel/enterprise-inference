"""Serving manifests rendered from templates/ ({{KEY}} placeholders)."""
from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Any

from . import platform
from .config import Catalog, Model, gateway_ref, mm_path, pick, text
from .ui import abort

CHAT_TEMPLATE_DIR = "/etc/chat-template"


def render(path: Path, **values: str) -> str:
    if not path.is_file():
        abort(f"Template not found: {path}")
    out = path.read_text(encoding="utf-8")
    for key, value in values.items():
        out = out.replace("{{" + key + "}}", value)
    return platform.adapt(out.rstrip("\n") + "\n")


def template(name: str) -> Path:
    return mm_path("MM_TEMPLATES", "templates") / name


def quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _list(items: list[str]) -> str:
    """Block-list items for a `key: {{X}}` placeholder; `[]` when empty (k8s rejects null)."""
    return "\n" + "\n".join(items) if items else "[]"


def inline_chat_template(m: Model) -> bool:
    """Inline Jinja ships in a ConfigMap: KServe runs the spec through Go templates and rejects {{ }}."""
    return bool(m.chat_template) and not m.chat_template.startswith("/")


def engine_args(m: Model) -> list[str]:
    """Category args < version delta < per-model args (CLI --arg is appended by the caller)."""
    return [text(a) for a in (*pick(m.cat, "args", []), *pick(m.ver, "args", []), *m.args)]


def trust_remote_code(m: Model, cli_args: list[str]) -> bool:
    return "--trust-remote-code" in (*engine_args(m), *cli_args)


def dedupe_args(args: list[str]) -> list[str]:
    """Keep the last occurrence of each flag (--k=v, --k v, --k), matching vLLM's last-wins parsing."""
    entries, i = [], 0
    while i < len(args):
        a = args[i]
        if re.match(r"--[^=]+=", a):
            entries.append((a.split("=", 1)[0], [a]))
        elif a.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            entries.append((a, [a, args[i + 1]]))
            i += 1
        else:
            entries.append((a, [a]))
        i += 1
    last = {key: n for n, (key, _) in enumerate(entries)}
    return [t for n, (key, tokens) in enumerate(entries) if last[key] == n for t in tokens]


def _env(m: Model, catalog: Catalog, cli_args: list[str], cli_env: list[str]) -> str:
    """Env, lowest to highest: trust-remote-code offline < runtime < category < version < model < --env."""
    merged = {"HF_HUB_OFFLINE": "1"} if trust_remote_code(m, cli_args) else {}
    for layer in (pick(m.rt, "env", {}), pick(m.cat, "env", {}), pick(m.ver, "env", {}), m.env,
                  dict((p.split("=", 1)[0], p.split("=", 1)[-1]) for p in cli_env)):
        merged.update(layer)
    kv_gb = text(pick(pick(m.cat, "defaults", {}), "kv_cache_gb", 8))
    items = [f"        - name: {k}\n          value: {quote(text(v).replace('${KV_CACHE_GB}', kv_gb))}"
             for k, v in sorted(merged.items())]
    if m.runtime == "openvino":  # OVMS fetches inside the serving pod, so it needs proxy + token there
        items += [f"        - name: {k}\n          value: {quote(v)}" for k, v in proxy_env(catalog)]
        if os.environ.get("HF_TOKEN"):
            items.append("        - name: HF_TOKEN\n          valueFrom:\n            secretKeyRef:\n"
                         "              name: hf-token\n              key: token")
    return _list(items)


def proxy_env(catalog: Catalog) -> list[tuple[str, str]]:
    pairs = (("HTTP_PROXY", catalog.http_proxy), ("HTTPS_PROXY", catalog.https_proxy), ("NO_PROXY", catalog.no_proxy))
    return [(k, v) for key, v in pairs if v for k in (key, key.lower())]


def _affinity(node: str, require_amx: bool) -> str:
    lines = ["    affinity:", "      nodeAffinity:"] if node or require_amx else []
    if require_amx:
        lines += ["        requiredDuringSchedulingIgnoredDuringExecution:", "          nodeSelectorTerms:",
                  "            - matchExpressions:", "                - key: feature.node.kubernetes.io/cpu-cpuid.AMXBF16",
                  "                  operator: In", "                  values:", '                    - "true"']
    if node:
        lines += ["        preferredDuringSchedulingIgnoredDuringExecution:", "          - weight: 100",
                  "            preference:", "              matchExpressions:",
                  "                - key: kubernetes.io/hostname", "                  operator: In",
                  "                  values:", f"                    - {node}", "          - weight: 50",
                  "            preference:", "              matchExpressions:",
                  "                - key: node-role.kubernetes.io/control-plane",
                  "                  operator: DoesNotExist"]
    return "\n".join(lines)


def _annotations(annotations: dict, kind: str) -> str:
    pad = "  " if kind == "LLMInferenceService" else "    "
    return f"{pad}annotations:\n" + "".join(
        f"{pad}  {k}: {quote(v) if v.isdigit() or v in ('true', 'false') else v}\n" for k, v in annotations.items())


# Standalone default; the install always sets MM_PROMETHEUS_URL from observability_namespace.
PROMETHEUS_URL = "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090"


def _llmisvc_replicas(m: Model) -> str:
    """spec.replicas, or spec.scaling with a KEDA trigger on the runtime's signal (KServe makes the
    ScaledObject). The CRD rejects both together. Model.check made both names DNS labels, safe in PromQL."""
    if not m.autoscaled:
        return f"  replicas: {m.replicas}"
    query = m.scale_query.replace("$namespace", m.namespace).replace("$model", m.name)
    return "\n".join([
        "  scaling:", f"    minReplicas: {m.min_replicas}", f"    maxReplicas: {m.max_replicas}", "    keda:",
        "      pollingInterval: 15", "      triggers:", "        - type: prometheus", "          metadata:",
        f"            serverAddress: {quote(os.environ.get('MM_PROMETHEUS_URL') or PROMETHEUS_URL)}",
        f"            query: {quote(query)}", f"            threshold: {quote(m.scale_target)}"])


def live_scaling(m: Model, spec: dict) -> Model:
    """m with the live spec's replica count or autoscaling range: the inverse of _llmisvc_replicas (and of an
    InferenceService's predictor.minReplicas)."""
    if m.kind != "LLMInferenceService":
        return dataclasses.replace(m, replicas=text(pick(pick(spec, "predictor", {}), "minReplicas", m.replicas)))
    scaling = pick(spec, "scaling", {})
    trigger = (pick(pick(scaling, "keda", {}), "triggers", []) or [{}])[0]
    return dataclasses.replace(m, replicas=text(pick(spec, "replicas", m.replicas)),
                               min_replicas=text(pick(scaling, "minReplicas")),
                               max_replicas=text(pick(scaling, "maxReplicas")),
                               scale_target=text(pick(pick(trigger, "metadata", {}), "threshold")))


def _fill(value: Any, m: Model) -> str:
    return text(value).replace("$model", m.name)


def _engine(m: Model) -> dict:
    """Command, probe paths and user of the main container: the runtime's own, or KServe's vLLM defaults."""
    c = m.container
    command = [_fill(a, m) for a in pick(c, "command", [])]
    return {"COMMAND": f"        command: {json.dumps(command)}\n" if command else "",
            "READY_PATH": _fill(pick(c, "ready_path", "/health"), m),
            "LIVE_PATH": _fill(pick(c, "live_path", "/health"), m), "RUN_AS": m.run_as_user}


def manifest(m: Model, catalog: Catalog, o, annotations: dict, cpu: str, cli_args: list[str]) -> str:
    """The LLMInferenceService / InferenceService for m. cli_args already carries any --tp injection."""
    args = engine_args(m)
    if m.kind == "LLMInferenceService" and m.container:
        if platform.is_openshift():  # OpenShift AI's template starts vLLM itself (see platform.adapt_llmisvc)
            abort(f"Runtime '{m.runtime}' is not supported on OpenShift: its LLMInferenceService template "
                  "starts vLLM.")
        if m.chat_template:
            abort(f"Model '{m.name}': chat_template applies to vLLM only, not runtime '{m.runtime}'.")
        args = [_fill(a, m) for a in pick(m.container, "args", [])] + args
    elif m.kind == "LLMInferenceService":
        args += ["--served-model-name", m.name]
        if m.chat_template:
            path = f"{CHAT_TEMPLATE_DIR}/chat_template.jinja" if inline_chat_template(m) else m.chat_template
            args += ["--chat-template", path]
    common = {
        "MODEL_NAME": m.name, "NAMESPACE": m.namespace, "MODEL_URI": m.uri, "CPU": cpu, "MEMORY": m.memory,
        "REPLICAS": m.replicas, "ARGS": _list([f"        - {quote(a)}" for a in dedupe_args(args + cli_args)]),
        "ENV": _env(m, catalog, cli_args, o.env), "NRI_ANNOTATIONS": _annotations(annotations, m.kind),
        "AFFINITY": _affinity(m.node, catalog.require_amx),
        "EXTRA_LABELS": "".join(f"\n    {p.split('=', 1)[0]}: {quote(p.split('=', 1)[1])}" for p in o.label),
    }
    if m.kind == "LLMInferenceService":
        inline = inline_chat_template(m)
        gateway, gateway_ns = gateway_ref()
        return platform.adapt_llmisvc(render(
            template("llm-inference-service.yaml"), **common, **_engine(m), IMAGE=m.image,
            SCALING=_llmisvc_replicas(m),
            GATEWAY_NAME=gateway, GATEWAY_NAMESPACE=gateway_ns,
            CHAT_TEMPLATE_MOUNT=(f"        volumeMounts:\n          - name: chat-template\n"
                                 f"            mountPath: {CHAT_TEMPLATE_DIR}\n            readOnly: true\n") if inline else "",
            CHAT_TEMPLATE_VOLUME=(f"    volumes:\n      - name: chat-template\n        configMap:\n"
                                  f"          name: {m.name}-chat-template\n") if inline else ""))
    if m.kind == "InferenceService":
        return render(template("inference-service.yaml"), **common, MODEL_ID=m.model_id, PVC_NAME=catalog.pvc,
                      MODEL_FORMAT=text(pick(m.rt, "format")) or "openvino",
                      RUNTIME_NAME=text(pick(m.rt, "runtime_name")) or f"{m.runtime}-runtime")
    abort(f"Unknown model kind: {m.kind}")
