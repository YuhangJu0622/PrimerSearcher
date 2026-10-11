"""Local trajectory artifacts for the OpenSeeker RL agent loop.

The VERL rollout output deliberately contains token tensors and scalar metrics,
not the full chat history. This module provides opt-in local artifacts for
debugging and offline trajectory curation without changing that contract.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def _enabled(value: str | None, *, default: bool = True) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def append_training_result(trace_id: str, event: str, **fields: Any) -> None:
    """Append a compact reward/training event to one rollout runtime log."""
    configured_directory = os.getenv("OPENSEEKER_AGENT_LOG_DIR")
    if not configured_directory or not trace_id:
        return
    if Path(trace_id).name != trace_id:
        return

    root = Path(configured_directory)
    root.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(fields, ensure_ascii=False, separators=(",", ":"), default=str)
    with (root / f"{trace_id}.log").open("a", encoding="utf-8") as handle:
        handle.write(f"TRAINING_RESULT event={event} payload={payload}\n")


class TrajectoryWriter:
    """Atomically write one completed rollout, when a directory is configured."""

    def __init__(self, directory: str | Path | None = None):
        configured_directory = directory or os.getenv("OPENSEEKER_TRAJECTORY_DIR")
        self.directory = Path(configured_directory) if configured_directory else None
        self.enabled = self.directory is not None and _enabled(
            os.getenv("OPENSEEKER_SAVE_TRAJECTORIES"), default=True
        )

    def write(
        self,
        trace_id: str,
        *,
        started_at: datetime,
        status: str,
        messages: list[dict[str, str]],
        extra_fields: dict[str, Any],
    ) -> Path | None:
        if not self.enabled or self.directory is None:
            return None

        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{trace_id}.json"
        temporary_path = self.directory / (
            f".{trace_id}.{os.getpid()}.{uuid4().hex}.json.tmp"
        )
        payload = {
            "schema_version": 1,
            "trace_id": trace_id,
            "pid": os.getpid(),
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "status": status,
            "messages": messages,
            "extra_fields": extra_fields,
        }
        try:
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"), default=str)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            return path
        finally:
            temporary_path.unlink(missing_ok=True)


class TraceLogger:
    """A per-rollout INFO/WARNING/ERROR log, mirrored to the process logger."""

    def __init__(self, trace_id: str, directory: str | Path | None = None):
        configured_directory = directory or os.getenv("OPENSEEKER_AGENT_LOG_DIR")
        self.trace_id = trace_id
        self._logger: logging.Logger | None = None
        self._handler: logging.Handler | None = None
        if not configured_directory:
            return

        root = Path(configured_directory)
        root.mkdir(parents=True, exist_ok=True)
        trace_logger = logging.getLogger(f"openseeker.rl.agent.{trace_id}")
        trace_logger.setLevel(
            getattr(logging, os.getenv("OPENSEEKER_AGENT_LOG_LEVEL", "INFO").upper(), logging.INFO)
        )
        trace_logger.propagate = False
        handler = logging.FileHandler(root / f"{trace_id}.log", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        trace_logger.addHandler(handler)
        self._logger = trace_logger
        self._handler = handler

    def log(self, level: int, message: str, *args: Any) -> None:
        logging.getLogger("src.openseeker_agent_loop").log(
            level, "[%s] " + message, self.trace_id, *args
        )
        if self._logger is not None:
            self._logger.log(level, message, *args)

    def debug(self, message: str, *args: Any) -> None:
        self.log(logging.DEBUG, message, *args)

    def info(self, message: str, *args: Any) -> None:
        self.log(logging.INFO, message, *args)

    def warning(self, message: str, *args: Any) -> None:
        self.log(logging.WARNING, message, *args)

    def error(self, message: str, *args: Any) -> None:
        self.log(logging.ERROR, message, *args)

    def close(self) -> None:
        if self._logger is not None and self._handler is not None:
            self._logger.removeHandler(self._handler)
            self._handler.close()
        self._handler = None
        self._logger = None
