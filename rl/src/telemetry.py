"""Low-overhead, content-free runtime telemetry for RL diagnostics."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "item"):
        return _safe(value.item())
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_safe(v) for v in value]
    return str(value)


class TelemetryWriter:
    """Append JSONL events without prompts, completions, tool inputs, or secrets."""

    def __init__(self, component: str):
        enabled = os.getenv("OPENSEEKER_TELEMETRY", "0").lower() in {"1", "true", "yes"}
        root = os.getenv("OPENSEEKER_TELEMETRY_DIR")
        self.directory = Path(root, component) if enabled and root else None
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def emit(self, trace_id: str, event: str, **fields: Any) -> None:
        if self.directory is None:
            return
        record = {
            "timestamp_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "event": event,
            **{key: _safe(value) for key, value in fields.items()},
        }
        path = self.directory / f"{trace_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
