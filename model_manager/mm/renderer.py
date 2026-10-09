import json
from pathlib import Path
from typing import Any

from mm.config import ModelConfig
from mm.ui import abort

class ManifestRenderer:
    def __init__(self, templates_dir: Path):
        self.templates_dir = templates_dir

    def _render(self, tmpl_name: str, vars_dict: dict[str, str]) -> str:
        tmpl_file = self.templates_dir / tmpl_name
        if not tmpl_file.is_file():
            abort(f"Template not found: {tmpl_file}")
        content = tmpl_file.read_text(encoding="utf-8")
        for k, v in vars_dict.items():
            content = content.replace(f"{{{{{k}}}}}", v)
        return content

    @staticmethod
    def dedupe_args(args_list: list[str]) -> list[str]:
        # Emulates vLLM last-wins CLI parsing
        # Items can be --flag=val, or --flag followed by val, or --flag
        entries: list[tuple[str, list[str]]] = []
        i = 0
        n = len(args_list)
        while i < n:
            token = args_list[i]
            if "=" in token and token.startswith("--"):
                flag_name = token.split("=")[0]
                entries.append((flag_name, [token]))
                i += 1
            elif token.startswith("--"):
                if i + 1 < n and not args_list[i + 1].startswith("--"):
                    entries.append((token, [token, args_list[i + 1]]))
                    i += 2
                else:
                    entries.append((token, [token]))
                    i += 1
            else:
                entries.append((token, [token]))
                i += 1

        seen = set()
        final_tokens: list[str] = []
        for key, tokens in reversed(entries):
            if key not in seen:
                seen.add(key)
                final_tokens = tokens + final_tokens
        return final_tokens

    def build_env_block(self, env_dict: dict[str, str], runtime: str, http_proxy: str, https_proxy: str, no_proxy: str, hf_token: str) -> str:
        out_lines = []
        for k, v in env_dict.items():
            out_lines.append(f"        - name: {k}\n          value: \"{v}\"")

        if runtime == "openvino":
            for k, v in [
                ("HTTP_PROXY", http_proxy), ("http_proxy", http_proxy),
                ("HTTPS_PROXY", https_proxy), ("https_proxy", https_proxy),
                ("NO_PROXY", no_proxy), ("no_proxy", no_proxy)
            ]:
                if v:
                    out_lines.append(f"        - name: {k}\n          value: \"{v}\"")
            if hf_token:
                out_lines.append(
                    "        - name: HF_TOKEN\n"
                    "          valueFrom:\n"
                    "            secretKeyRef:\n"
                    "              name: hf-token\n"
                    "              key: token"
                )

        if not out_lines:
            return "[]"
        return "\n" + "\n".join(out_lines)

    def build_args_block(self, args_list: list[str]) -> str:
        if not args_list:
            return "[]"
        deduped = self.dedupe_args(args_list)
        out_lines = []
        for a in deduped:
            out_lines.append(f"        - {json.dumps(a)}")
        return "\n" + "\n".join(out_lines)

    def build_labels_block(self, extra_labels: dict[str, str]) -> str:
        if not extra_labels:
            return ""
        out = ""
        for k, v in extra_labels.items():
            out += f"\n    {k}: {json.dumps(v)}"
        return out

    def build_affinity_block(self, node: str = "", require_amx: bool = True) -> str:
        if not node and not require_amx:
            return ""

        lines = [
            "    affinity:",
            "      nodeAffinity:"
        ]
        if require_amx:
            lines.extend([
                "        requiredDuringSchedulingIgnoredDuringExecution:",
                "          nodeSelectorTerms:",
                "            - matchExpressions:",
                "                - key: feature.node.kubernetes.io/cpu-cpuid.AMXBF16",
                "                  operator: In",
                "                  values:",
                '                    - "true"'
            ])

        if node:
            lines.extend([
                "        preferredDuringSchedulingIgnoredDuringExecution:",
                "          - weight: 100",
                "            preference:",
                "              matchExpressions:",
                "                - key: kubernetes.io/hostname",
                "                  operator: In",
                "                  values:",
                f"                    - {node}",
                "          - weight: 50",
                "            preference:",
                "              matchExpressions:",
                "                - key: node-role.kubernetes.io/control-plane",
                "                  operator: DoesNotExist"
            ])
        return "\n".join(lines)

    def build_annotations_block(self, annotations: dict[str, str], indent: str = "    ") -> str:
        if not annotations:
            return ""
        lines = [f"{indent}annotations:"]
        for k, v in annotations.items():
            lines.append(f"{indent}  {k}: {v}")
        return "\n".join(lines) + "\n"
