"""Model catalog (models.yaml) and resolution of a catalog entry or --id into a Model."""
from __future__ import annotations

import json
import os
import re
import subprocess  # nosec B404
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from .ui import abort, info

MM_DIR = Path(__file__).resolve().parents[1]
DNS_LABEL = re.compile(r"[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?")
HF_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9._-]+)?")
PIPELINE_CATEGORY = {
    "text-generation": "llm", "text2text-generation": "llm", "conversational": "llm",
    "feature-extraction": "embed", "sentence-similarity": "embed", "fill-mask": "embed",
    "text-classification": "rerank",
    "image-text-to-text": "vlm", "visual-question-answering": "vlm", "image-to-text": "vlm",
}


def mm_path(env: str, default: str) -> Path:
    return Path(os.environ.get(env) or MM_DIR / default)


def text(value: Any) -> str:
    """A YAML value as text, the way `jq -r` prints it (true, 8, 1.5)."""
    return value if isinstance(value, str) else json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def is_uint(value: str) -> bool:
    return re.fullmatch(r"[0-9]+", value) is not None


def truthy(value: Any) -> bool:
    return text(value).lower() in ("true", "1", "yes")


def pick(d: Any, key: str, default: Any = "") -> Any:
    """jq `.key // default`: a missing, null or false value falls back to the default."""
    v = d.get(key) if isinstance(d, dict) else None
    return default if v is None or v is False else v


def shaped(what: str, value: Any, want: type, source: Path) -> Any:
    if isinstance(value, want):
        return value
    names = {dict: "object", list: "array", str: "string", bool: "boolean", type(None): "null"}
    got = names.get(type(value), "number")
    hint = "a list of strings" if want is list else "a mapping of KEY: VALUE"
    abort(f"Invalid {what}: expected {names[want]} ({hint}) in {source}, got {got}.")


def load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        abort(f"Cannot read {path}: {e}")


def hf_model_info(model_id: str, proxy: str) -> dict:
    """HuggingFace model metadata, or {} when the Hub is unreachable."""
    cmd = ["curl", "-sf", "--max-time", "10", *(["--proxy", proxy] if proxy else []),
           f"https://huggingface.co/api/models/{model_id}"]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)  # nosec B603
    try:
        data = json.loads(r.stdout) if r.returncode == 0 else {}
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


class Catalog:
    def __init__(self, path: Path):
        if not path.is_file():
            abort(f"Config not found: {path}")
        self.path = path
        self.data = load_yaml(path) or {}
        d = pick(self.data, "defaults", {})
        self.namespace = text(pick(d, "namespace", "llm-inference"))
        self.runtime = text(pick(d, "runtime", "vllm"))
        self.replicas = text(pick(d, "replicas", 1))
        self.require_amx = truthy(pick(d, "require_amx", False))
        self.pvc = text(pick(pick(self.data, "storage", {}), "pvc_name", "model-store"))
        net = pick(self.data, "network", {})
        self.http_proxy, self.https_proxy, self.no_proxy = (
            text(pick(net, f"{p}_proxy")) or os.environ.get(f"{p.upper()}_PROXY") or os.environ.get(f"{p}_proxy") or ""
            for p in ("http", "https", "no"))
        self.connectivity_url = text(pick(net, "connectivity_check", "https://huggingface.co"))
        self.models = [m for m in pick(self.data, "models", []) if isinstance(m, dict)]

    @classmethod
    def load(cls) -> Catalog:
        return cls(mm_path("MM_CONFIG", "models.yaml"))

    def names(self) -> list[str]:
        return [text(pick(m, "name")) for m in self.models]

    def entry(self, name: str) -> dict | None:
        return next((m for m in self.models if pick(m, "name") == name), None)

    def runtime_config(self, name: str) -> dict:
        """runtimes/<name>/values.yaml deep-merged with models.yaml runtimes.<name> (models.yaml wins)."""
        values = mm_path("MM_RUNTIMES", "runtimes") / name / "values.yaml"
        base = (load_yaml(values) if values.is_file() else None) or {}
        overlay = pick(pick(self.data, "runtimes", {}), name, {})
        if not base and not overlay:
            abort(f"Runtime '{name}' not found under 'runtimes:' in {self.path} or at {values}")
        return _merge(base, overlay)


def _merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        out[k] = _merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


@dataclass
class Model:
    name: str
    namespace: str
    kind: str = ""
    model_id: str = ""
    runtime: str = ""
    rt: dict = field(default_factory=dict)    # merged runtime config
    ver: dict = field(default_factory=dict)   # selected runtime version entry (env/args deltas)
    version: str = ""                         # "" for runtimes without a versions: map
    runtime_image: str = ""
    image: str = ""
    category: str = ""
    routing: str = ""
    cpu: str = ""
    memory: str = ""
    replicas: str = "1"
    node: str = ""
    chat_template: str = ""
    args: list = field(default_factory=list)
    env: dict = field(default_factory=dict)
    uri: str = ""

    @property
    def cat(self) -> dict:
        return pick(pick(self.rt, "categories", {}), self.category or "llm", {})

    def check(self) -> None:
        """Reject names/ids that are not safe to embed in manifests and the download script."""
        if not DNS_LABEL.fullmatch(self.name):
            abort(f"Invalid model name '{self.name}': use lowercase letters, digits and '-' (max 63 characters).")
        if not HF_REPO.fullmatch(self.model_id):
            abort(f"Invalid model_id '{self.model_id}' for '{self.name}': expected a HuggingFace repo id "
                  "such as org/name.")


def resolve(catalog: Catalog, target: str, o: Any) -> Model:
    """Resolve a catalog model (or --id) plus CLI overrides into the Model to deploy."""
    if o.id:
        name = o.name or re.sub(r"[^a-zA-Z0-9]", "-", o.id.rsplit("/", 1)[-1]).lower()
        m = Model(name=name, namespace=catalog.namespace, model_id=o.id, runtime=o.server or catalog.runtime,
                  version=o.server_version, replicas=catalog.replicas)
    elif target:
        m = _from_catalog(catalog, target, o)
    else:
        abort("No model specified. Use: model-manager <command> <model-name> or --id <hf/repo>")

    m.rt = catalog.runtime_config(m.runtime)
    m.kind = m.kind or text(pick(m.rt, "kind", "LLMInferenceService"))
    _select_image(catalog, m, o)

    explicit_routing = bool(m.routing or o.routing)
    m.category = o.category or m.category
    if not m.category:
        m.category = PIPELINE_CATEGORY.get(
            text(pick(hf_model_info(m.model_id, catalog.https_proxy), "pipeline_tag")), "llm") if m.model_id else "llm"
        info(f"Category: {m.category} (auto-detected)")
    what = f"runtime '{m.runtime}'"
    shaped(f"{what} env", pick(m.rt, "env", {}), dict, catalog.path)
    shaped(f"{what} category '{m.category}' args", pick(m.cat, "args", []), list, catalog.path)
    shaped(f"{what} category '{m.category}' env", pick(m.cat, "env", {}), dict, catalog.path)

    defaults = pick(m.cat, "defaults", {})
    m.cpu = o.cpu or m.cpu or text(pick(defaults, "cpu", 8))
    m.memory = o.memory or m.memory or text(pick(defaults, "memory", "16Gi"))
    m.replicas = o.replicas or m.replicas
    m.node = o.node or m.node
    if o.routing:
        m.routing = o.routing
    elif not explicit_routing and m.category in ("embed", "rerank"):
        m.routing = "direct"  # no KV cache, so EPP's cache-aware routing buys nothing
    elif not m.routing:
        m.routing = text(pick(m.rt, "routing"))
    m.image = m.image or m.runtime_image
    m.name = o.alias or m.name
    m.uri = f"pvc://{catalog.pvc}/{m.name}"
    return m


def _from_catalog(catalog: Catalog, name: str, o: Any) -> Model:
    e = catalog.entry(name)
    if e is None:
        abort(f"Model '{name}' not found in {catalog.path}")
    template = e.get("chat_template")
    if isinstance(template, (list, dict)):
        abort(f"Invalid model '{name}' chat_template: expected a string (inline Jinja or a file path) in "
              f"{catalog.path}, got {'array' if isinstance(template, list) else 'object'}. Some HuggingFace repos "
              'publish a list of named templates — copy the single template you want (e.g. the one named "default") '
              "as a block scalar.")
    m = Model(name=name, namespace=text(pick(e, "namespace", catalog.namespace)), kind=text(pick(e, "kind")),
              model_id=text(pick(e, "model_id")), cpu=text(pick(e, "cpu")), memory=text(pick(e, "memory")),
              replicas=text(pick(e, "replicas", catalog.replicas)), category=text(pick(e, "category")),
              image=text(pick(e, "image")), node=text(pick(e, "node")), routing=text(pick(e, "routing")),
              chat_template=text(pick(e, "chat_template")).rstrip("\n"),  # as legacy $(jq) did
              env=shaped(f"model '{name}' env", pick(e, "env", {}), dict, catalog.path),
              args=shaped(f"model '{name}' args", pick(e, "args", []), list, catalog.path))

    # Server binding: a servers: map is an allow-list; runtime:/server_version: is shorthand without one.
    servers = shaped(f"model '{name}' servers", pick(e, "servers", {}), dict, catalog.path)
    explicit = bool(servers)
    default_server = text(pick(e, "default_server"))
    if not explicit:
        default_server = text(pick(e, "runtime")) or catalog.runtime
        servers = {default_server: {"default": text(pick(e, "server_version"))}}
    m.runtime = o.server or default_server or min(servers, default="") or catalog.runtime
    if explicit and m.runtime not in servers:
        abort(f"Model '{name}' does not declare server '{m.runtime}'. Declared: {', '.join(sorted(servers))}")
    entry = pick(servers, m.runtime, {})
    declared = [text(v) for v in pick(entry, "versions", [])]
    m.version = o.server_version or text(pick(entry, "default", declared[0] if declared else ""))
    if declared and m.version and m.version not in declared:
        abort(f"Model '{name}' is not declared to support '{m.runtime}' version '{m.version}'. "
              f"Declared versions: {', '.join(declared)}")
    return m


def _select_image(catalog: Catalog, m: Model, o: Any) -> None:
    """Pin m.version to a runtime version and take its image; flat runtimes have a single image."""
    if "versions" not in m.rt:
        if o.server_version:
            abort(f"Runtime '{m.runtime}' does not define a 'versions:' map, so --server-version "
                  f"'{o.server_version}' cannot be selected.")
        m.version, m.runtime_image = "", text(m.rt.get("image"))
        return
    versions = {text(k): v for k, v in (m.rt["versions"] or {}).items()}
    m.version = m.version or text(pick(m.rt, "default_version")) or min(versions, default="")
    m.ver = pick(versions, m.version, None)
    if m.ver is None:
        abort(f"Runtime '{m.runtime}' has no version '{m.version}'. Available: {', '.join(sorted(versions))}")
    m.runtime_image = text(pick(m.ver, "image"))
    if not m.runtime_image:
        abort(f"Runtime '{m.runtime}' version '{m.version}' defines no image")
    what = f"runtime '{m.runtime}' version '{m.version}'"
    shaped(f"{what} env", pick(m.ver, "env", {}), dict, catalog.path)
    shaped(f"{what} args", pick(m.ver, "args", []), list, catalog.path)


def _repo_roots() -> list[Path]:
    """Where to look for solution configs: MM_GLOBAL_CONFIG_SEARCH, the enclosing repo, sibling checkouts."""
    roots = [Path(p) for p in (os.environ.get("MM_GLOBAL_CONFIG_SEARCH") or "").split(":") if p]
    roots += [d for d in (MM_DIR, *list(MM_DIR.parents)[:-1]) if (d / "es_auto_installer.sh").is_file()][:1]
    return roots + sorted(p for p in MM_DIR.parent.parent.glob("*ai-solutions*") if p.is_dir())


def find_config(env: str, *subpaths: str) -> Path | None:
    p = os.environ.get(env)
    if p and Path(p).is_file():
        return Path(p)
    return next((r / s for r in _repo_roots() for s in subpaths if (r / s).is_file()), None)


def global_config() -> Path | None:
    return find_config("MM_GLOBAL_CONFIG", "env/local/global_config.yaml", "config/defaults/global_config.yaml")


def inference_config() -> Path | None:
    return find_config("MM_INFERENCE_CONFIG", "env/local/config.inference.yaml", "ext/enterprise.ai-inference/config.yaml")


@cache
def _config_data(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def config_value(path: Path | None, key: str) -> str:
    return text(pick(_config_data(path), key)) if path else ""
