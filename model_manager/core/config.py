from dataclasses import dataclass, field
from pathlib import Path
import os
import sys
from typing import Any
import yaml

from .ui import abort

@dataclass
class DefaultsConfig:
    namespace: str = "llm-inference"
    runtime: str = "vllm"
    replicas: int = 1
    require_amx: bool = False

@dataclass
class StorageConfig:
    pvc_name: str = "model-store"

@dataclass
class NetworkConfig:
    http_proxy: str = ""
    https_proxy: str = ""
    no_proxy: str = ""
    connectivity_check: str = "https://huggingface.co"

@dataclass
class ModelConfig:
    name: str
    model_id: str
    category: str = ""
    kind: str = ""
    namespace: str = ""
    cpu: str | int = ""
    memory: str = ""
    replicas: int = 1
    image: str = ""
    node: str = ""
    routing: str = ""
    runtime: str = ""
    server_version: str = ""
    servers: dict[str, Any] = field(default_factory=dict)
    default_server: str = ""
    chat_template: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    extra_labels: dict[str, str] = field(default_factory=dict)

@dataclass
class AppConfig:
    defaults: DefaultsConfig
    storage: StorageConfig
    network: NetworkConfig
    models: dict[str, ModelConfig]
    runtimes: dict[str, Any]
    raw_config: dict[str, Any]

def load_app_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    if not path.is_file():
        abort(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_defs = data.get("defaults", {})
    defaults = DefaultsConfig(
        namespace=raw_defs.get("namespace", "llm-inference"),
        runtime=raw_defs.get("runtime", "vllm"),
        replicas=int(raw_defs.get("replicas", 1)),
        require_amx=bool(raw_defs.get("require_amx", False)),
    )

    raw_st = data.get("storage", {})
    storage = StorageConfig(
        pvc_name=raw_st.get("pvc_name", "model-store")
    )

    raw_net = data.get("network", {})
    network = NetworkConfig(
        http_proxy=raw_net.get("http_proxy") or os.getenv("HTTP_PROXY") or os.getenv("http_proxy") or "",
        https_proxy=raw_net.get("https_proxy") or os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or "",
        no_proxy=raw_net.get("no_proxy") or os.getenv("NO_PROXY") or os.getenv("no_proxy") or "",
        connectivity_check=raw_net.get("connectivity_check", "https://huggingface.co"),
    )

    models: dict[str, ModelConfig] = {}
    for m in data.get("models", []):
        name = m.get("name")
        if not name:
            continue
        models[name] = ModelConfig(
            name=name,
            model_id=m.get("model_id", ""),
            category=m.get("category", ""),
            kind=m.get("kind", ""),
            namespace=m.get("namespace", defaults.namespace),
            cpu=m.get("cpu", ""),
            memory=str(m.get("memory", "")) if m.get("memory") is not None else "",
            replicas=int(m.get("replicas", defaults.replicas)),
            image=m.get("image", ""),
            node=m.get("node", ""),
            routing=m.get("routing", ""),
            runtime=m.get("runtime", ""),
            server_version=str(m.get("server_version", "")) if m.get("server_version") is not None else "",
            servers=m.get("servers") or {},
            default_server=m.get("default_server", ""),
            chat_template=m.get("chat_template") or "",
            args=m.get("args") or [],
            env=m.get("env") or {},
        )

    runtimes = data.get("runtimes", {})
    return AppConfig(defaults, storage, network, models, runtimes, data)

def _discover_repo_roots(script_dir: Path) -> list[Path]:
    roots: list[Path] = []
    if os.getenv("MM_GLOBAL_CONFIG_SEARCH"):
        for p in os.environ["MM_GLOBAL_CONFIG_SEARCH"].split(":"):
            candidate = Path(p.strip())
            if candidate.is_dir():
                roots.append(candidate)

    # Search upwards for repo root containing es_auto_installer.sh
    cur = script_dir.resolve()
    while cur != cur.parent:
        if (cur / "es_auto_installer.sh").is_file():
            if cur not in roots:
                roots.append(cur)
            break
        cur = cur.parent

    # Siblings matching *ai-solutions*
    if roots:
        parent = roots[0].parent
        for s in parent.glob("*ai-solutions*"):
            if s.is_dir() and s not in roots:
                roots.append(s)

    return roots

def discover_global_config(script_dir: Path) -> Path | None:
    if os.getenv("MM_GLOBAL_CONFIG") and Path(os.environ["MM_GLOBAL_CONFIG"]).is_file():
        return Path(os.environ["MM_GLOBAL_CONFIG"])

    roots = _discover_repo_roots(script_dir)
    sub_paths = ["env/local/global_config.yaml", "config/defaults/global_config.yaml"]
    for r in roots:
        for sub in sub_paths:
            candidate = r / sub
            if candidate.is_file():
                return candidate
    return None

def discover_inference_config(script_dir: Path) -> Path | None:
    if os.getenv("MM_INFERENCE_CONFIG") and Path(os.environ["MM_INFERENCE_CONFIG"]).is_file():
        return Path(os.environ["MM_INFERENCE_CONFIG"])

    roots = _discover_repo_roots(script_dir)
    sub_paths = ["env/local/config.inference.yaml", "ext/enterprise.ai-inference/config.yaml"]
    for r in roots:
        for sub in sub_paths:
            candidate = r / sub
            if candidate.is_file():
                return candidate
    return None
