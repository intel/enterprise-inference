"""Unit tests for behaviour the parity suite cannot observe, mostly fixes over the legacy CLI."""
from __future__ import annotations

import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import cli, download, k8s, routing  # noqa: E402
from core.config import Catalog, Model  # noqa: E402
from core.policy import _whole_cpus  # noqa: E402
from core.renderer import dedupe_args, engine_args  # noqa: E402


@pytest.fixture
def kubectl(monkeypatch):
    """Record kubectl calls as (argv, stdin); every call succeeds with empty output."""
    calls = []

    def run(*args, stdin=None):
        calls.append((args, stdin))
        return subprocess.CompletedProcess(args, 0, "", "")
    monkeypatch.setattr(k8s, "run", run)
    return calls


def test_arg_values_taken_verbatim():
    # Legacy dropped the first --arg (jq without -n); argparse would reject a value starting with --.
    o = cli.parse_args(["deploy", "m", "--arg", "--max-model-len=4096", "--arg", "--enforce-eager"])
    assert o.arg == ["--max-model-len=4096", "--enforce-eager"] and o.positional == ["deploy", "m"]


@pytest.mark.parametrize("args, tp", [(["--tensor-parallel-size", "2"], 2), (["--tensor-parallel-size=4"], 4), ([], 1)])
def test_tp_from_engine_args(args, tp):
    assert cli._engine_cli_args(cli.Options(arg=args)) == (args, tp)


def test_tp_flag_is_injected_once():
    assert cli._engine_cli_args(cli.Options(tp="2")) == (["--tensor-parallel-size", "2"], 2)


def test_dedupe_keeps_last_occurrence():
    assert dedupe_args(["--a", "1", "--b", "--a=2", "x"]) == ["--b", "--a=2", "x"]


def test_numeric_catalog_args_are_rendered():
    # Legacy jq failed on a YAML number and rendered `args: []`.
    assert engine_args(Model(name="m", namespace="n", args=["--max-model-len", 2048])) == ["--max-model-len", "2048"]


@pytest.mark.parametrize("cpu, whole", [("8", 8), ("7.5", 8), ("7500m", 8), ("0m", 0), ("abc", 0), ("²", 0)])
def test_whole_cpus(cpu, whole):
    assert _whole_cpus(cpu) == whole


@pytest.mark.parametrize("name, model_id", [("ok", "org/m'; rm -rf /models #"), ("Bad_Name", "org/m")])
def test_unsafe_names_rejected(name, model_id):
    with pytest.raises(SystemExit):
        Model(name=name, namespace="n", model_id=model_id).check()


def test_hf_token_only_on_stdin(kubectl, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setattr(download, "_curl_ok", lambda *_: True)
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    m = Model(name="m", namespace="n", model_id="org/m", runtime="vllm", runtime_image="img", cpu="8", memory="8Gi")
    download.download_weights(m, Catalog(cli.MM_DIR / "models.yaml"), [])
    assert not any("hf_secret" in " ".join(args) for args, _ in kubectl)
    assert any(stdin == "hf_secret" for _, stdin in kubectl)


def test_litellm_key_only_on_stdin(kubectl, monkeypatch):
    monkeypatch.setattr(routing, "_litellm_key", lambda: "sk-master")
    routing._litellm("GET", "http://litellm/model/info")
    (args, stdin), = kubectl
    assert "sk-master" not in " ".join(args) and stdin == "Authorization: Bearer sk-master\n"


def test_cpu_collisions_check_runs_report(monkeypatch):
    seen = []
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, **_: seen.append(cmd) or subprocess.CompletedProcess(cmd, 3))
    monkeypatch.setenv("MM_CONFIG", str(cli.MM_DIR / "models.yaml"))
    with pytest.raises(SystemExit) as exit_:
        cli.cpu_collisions_check(cli.Options(), "")
    assert exit_.value.code == 3 and seen[0][1:] == [str(cli.MM_DIR / "scripts" / "nri-cpu-report.py"),
                                                       "--namespace", "llm-inference"]
