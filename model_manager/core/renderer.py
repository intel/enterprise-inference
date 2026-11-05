"""Serving manifests rendered from templates/ ({{KEY}} placeholders)."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

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


def manifest(m: Model, catalog: Catalog, o, annotations: dict, cpu: str, cli_args: list[str]) -> str:
    """The LLMInferenceService / InferenceService for m. cli_args already carries any --tp injection."""
    args = engine_args(m)
    if m.kind == "LLMInferenceService":
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
            template("llm-inference-service.yaml"), **common, IMAGE=m.image,
            GATEWAY_NAME=gateway, GATEWAY_NAMESPACE=gateway_ns,
            CHAT_TEMPLATE_MOUNT=(f"        volumeMounts:\n          - name: chat-template\n"
                                 f"            mountPath: {CHAT_TEMPLATE_DIR}\n            readOnly: true\n") if inline else "",
            CHAT_TEMPLATE_VOLUME=(f"    volumes:\n      - name: chat-template\n        configMap:\n"
                                  f"          name: {m.name}-chat-template\n") if inline else ""))
    if m.kind == "InferenceService":
        name = "inference-service-ovms.yaml" if m.runtime == "openvino" else "inference-service.yaml"
        return render(template(name), **common, MODEL_ID=m.model_id, PVC_NAME=catalog.pvc,
                      MODEL_FORMAT=text(pick(m.rt, "format")) or "openvino",
                      RUNTIME_NAME=text(pick(m.rt, "runtime_name")) or f"{m.runtime}-runtime")
    abort(f"Unknown model kind: {m.kind}")
