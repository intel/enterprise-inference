import sys

# ANSI Escape Sequences
_USE_COLOR = sys.stdout.isatty()

BOLD = "\033[1m" if _USE_COLOR else ""
RED = "\033[1;31m" if _USE_COLOR else ""
GREEN = "\033[1;32m" if _USE_COLOR else ""
YELLOW = "\033[1;33m" if _USE_COLOR else ""
BLUE = "\033[1;34m" if _USE_COLOR else ""
CYAN = "\033[1;36m" if _USE_COLOR else ""
DIM = "\033[90m" if _USE_COLOR else ""
RESET = "\033[0m" if _USE_COLOR else ""

def info(msg: str) -> None:
    print(f"{BLUE}[INFO ]{RESET}  {msg}")

def ok(msg: str) -> None:
    print(f"{GREEN}[  OK ]{RESET}  {msg}")

def warn(msg: str) -> None:
    print(f"{YELLOW}[WARN ]{RESET}  {msg}", file=sys.stderr)

def err(msg: str) -> None:
    print(f"{RED}[ERROR]{RESET}  {msg}", file=sys.stderr)

def abort(*msgs: str) -> None:
    for m in msgs:
        err(m)
    sys.exit(1)

def fmt_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60}s"
