"""Console output: colored status lines (info/ok on stdout, warn/err on stderr)."""
from __future__ import annotations

import sys

_COLOR = sys.stdout.isatty()


def _c(code: str) -> str:
    return f"\033[{code}m" if _COLOR else ""


BOLD, RED, GREEN, YELLOW, BLUE, CYAN, DIM, RESET = (
    _c(c) for c in ("1", "1;31", "1;32", "1;33", "1;34", "1;36", "90", "0"))


def info(msg: str, err: bool = False) -> None:
    print(f"{BLUE}[INFO ]{RESET}  {msg}", file=sys.stderr if err else sys.stdout)


def ok(msg: str) -> None:
    print(f"{GREEN}[  OK ]{RESET}  {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}[WARN ]{RESET}  {msg}", file=sys.stderr)


def abort(msg: str):
    print(f"{RED}[ERROR]{RESET}  {msg}", file=sys.stderr)
    sys.exit(1)


def fmt_duration(seconds: int) -> str:
    return f"{seconds}s" if seconds < 60 else f"{seconds // 60}m {seconds % 60}s"
