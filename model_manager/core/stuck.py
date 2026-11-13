"""Pod states that will not clear on their own, so a wait stops with the reason instead of running out its timeout."""
from __future__ import annotations

import os
import time

STUCK_WAITING = {"ImagePullBackOff", "ErrImagePull", "InvalidImageName", "CreateContainerConfigError",
                 "CreateContainerError", "RunContainerError"}
CRASH_RESTARTS = 3


def grace() -> int:
    """Seconds a state must last: the scheduler retries as other pods free their resources."""
    value = os.environ.get("MM_STUCK_GRACE", "")
    return int(value) if value.isdigit() else 180


def blocker(pod: dict) -> tuple[str, bool]:
    """(why the pod cannot run, whether that is final at once); ('', False) when it can, or still may."""
    status = pod.get("status") or {}
    for c in status.get("conditions") or []:
        if c.get("type") == "PodScheduled" and c.get("status") == "False" and c.get("reason") == "Unschedulable":
            return f"cannot be scheduled: {c.get('message', '').strip()}", False
    for cs in (status.get("initContainerStatuses") or []) + (status.get("containerStatuses") or []):
        waiting = (cs.get("state") or {}).get("waiting") or {}
        reason = waiting.get("reason", "")
        if reason in STUCK_WAITING:
            return f"container {cs.get('name')}: {reason}: {waiting.get('message', '').strip()}", False
        if reason == "CrashLoopBackOff" and cs.get("restartCount", 0) >= CRASH_RESTARTS:
            last = (cs.get("lastState") or {}).get("terminated") or {}
            return (f"container {cs.get('name')} keeps crashing ({cs.get('restartCount')} restarts, last exit "
                    f"{last.get('exitCode', '?')} {last.get('reason', '')}".rstrip() + ")"), True
    return "", False


class Watch:
    """Reports a pod blocker once it has lasted grace() seconds (a crash loop at once)."""

    def __init__(self) -> None:
        self.since: dict[str, float] = {}

    def stuck(self, pods: list[dict]) -> str:
        now, seen = time.monotonic(), {}
        for pod in pods:
            reason, final = blocker(pod)
            if not reason:
                continue
            name = (pod.get("metadata") or {}).get("name", "?")
            key = f"{name}|{reason.split(':', 1)[0]}"
            seen[key] = self.since.get(key, now)
            if final or now - seen[key] >= grace():
                return f"pod {name} {reason}"
        self.since = seen
        return ""
