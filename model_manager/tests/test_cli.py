"""Tests for CLI parsing and validation in model_manager.core.cli."""
import pytest
from model_manager.core.cli import require_label, build_parser

def test_require_label_valid():
    require_label("app=frontend")
    require_label("example.com/tier=cache")
    require_label("flag=")
    require_label("env=prod-1")

def test_require_label_invalid():
    with pytest.raises(SystemExit):
        require_label("no_equals_sign")
    with pytest.raises(SystemExit):
        require_label("=value_only")
    with pytest.raises(SystemExit):
        require_label("invalid key=value")

def test_parser_deploy_subcommand():
    parser = build_parser()
    args = parser.parse_args(["deploy", "qwen3-0-6b", "--wait", "--dry-run", "--cpu", "16"])
    assert args.command == "deploy"
    assert args.model == "qwen3-0-6b"
    assert args.wait is True
    assert args.dry_run is True
    assert args.cpu == "16"

def test_parser_undeploy_subcommand():
    parser = build_parser()
    args = parser.parse_args(["undeploy", "llama3-8b-awq", "--wait-timeout", "120"])
    assert args.command == "undeploy"
    assert args.model == "llama3-8b-awq"
    assert args.wait_timeout == 120

def test_parser_cpu_collisions_check():
    parser = build_parser()
    args = parser.parse_args(["cpu-collisions-check"])
    assert args.command == "cpu-collisions-check"

def test_cli_subcommands_and_order(monkeypatch):
    import subprocess
    import sys
    from pathlib import Path

    root_dir = Path(__file__).resolve().parent.parent
    py_exe = sys.executable

    # Test unknown option exits with code 1
    res = subprocess.run([py_exe, "-m", "core", "--bad-flag"], cwd=str(root_dir), capture_output=True, text=True)
    assert res.returncode == 1
    assert "Unknown option: --bad-flag" in res.stderr

    # Test unknown command exits with code 1
    res = subprocess.run([py_exe, "-m", "core", "nonexistent-cmd"], cwd=str(root_dir), capture_output=True, text=True)
    assert res.returncode == 1
    assert "Unknown command: nonexistent-cmd" in res.stderr

    # Test options before subcommand work
    res = subprocess.run([py_exe, "-m", "core", "--dry-run", "deploy", "qwen3-0-6b"], cwd=str(root_dir), capture_output=True, text=True)
    assert res.returncode == 0
    assert "LLMInferenceService" in res.stdout

    # Test version subcommand
    res = subprocess.run([py_exe, "-m", "core", "version"], cwd=str(root_dir), capture_output=True, text=True)
    assert res.returncode == 0
    assert "model-manager 2.1.0" in res.stdout

    # Test help subcommand
    res = subprocess.run([py_exe, "-m", "core", "help"], cwd=str(root_dir), capture_output=True, text=True)
    assert res.returncode == 0
    assert "model-manager — Deploy LLM inference models on Kubernetes" in res.stdout

