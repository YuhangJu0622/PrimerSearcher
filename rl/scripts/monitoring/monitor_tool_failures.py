#!/usr/bin/env python3
"""Monitor RL agent logs and send an AutoDL WeChat alert on terminal tool failures.

Only lines emitted by ``tool_runtime.py`` in this stable form are considered:

    TOOL_TERMINAL outcome=success|failure tool=search|visit ...

A success resets the streak.  A failure is deliberately an AgentLoop-recoverable
condition, but a sustained streak normally means the external tool environment
is unhealthy and should halt or investigate training.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable

import requests


TERMINAL_EVENT_RE = re.compile(
    r"TOOL_TERMINAL\s+outcome=(?P<outcome>success|failure)\s+tool=(?P<tool>search|visit)\b"
)
STATE_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SECRETS_FILE = PROJECT_ROOT / ".secrets" / "tool_api_keys.env"


def load_autodl_secret_from_file(path: Path) -> None:
    """Load only the monitor token/name from the existing mode-600 secrets file."""
    if not path.exists():
        return
    if path.stat().st_mode & 0o777 != 0o600:
        raise RuntimeError(f"Refusing secrets file without 600 permissions: {path}")
    externally_configured = {
        key
        for key in ("OPENSEEKER_AUTODL_TOKEN", "OPENSEEKER_AUTODL_MESSAGE_NAME")
        if key in os.environ
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key not in {"OPENSEEKER_AUTODL_TOKEN", "OPENSEEKER_AUTODL_MESSAGE_NAME"}:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if value and key not in externally_configured:
            # Match shell `source`: a later line in the secrets file wins.
            os.environ[key] = value


AUTODL_WECHAT_URL = "https://www.autodl.com/api/v1/wechat/message/send"


class ToolFailureMonitor:
    """Tails per-rollout logs without rereading previously handled events."""

    def __init__(
        self,
        *,
        log_dir: Path,
        state_file: Path,
        threshold: int,
        require_both_tools: bool,
        start_at_end: bool,
        alert_callback: Callable[[dict], bool],
        logger: logging.Logger,
    ) -> None:
        self.log_dir = log_dir
        self.state_file = state_file
        self.threshold = threshold
        self.require_both_tools = require_both_tools
        self.start_at_end = start_at_end
        self.alert_callback = alert_callback
        self.logger = logger
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if not self.state_file.exists():
            return {
                "version": STATE_VERSION,
                "initialized": False,
                "files": {},
                "consecutive_failures": 0,
                "failure_counts": {"search": 0, "visit": 0},
                "alerted_in_current_streak": False,
                "recent_failures": [],
            }
        try:
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
            if state.get("version") != STATE_VERSION:
                raise ValueError("unsupported state version")
            return state
        except Exception as error:
            self.logger.warning("Ignoring unreadable monitor state %s: %s", self.state_file, error)
            return {
                "version": STATE_VERSION,
                "initialized": False,
                "files": {},
                "consecutive_failures": 0,
                "failure_counts": {"search": 0, "visit": 0},
                "alerted_in_current_streak": False,
                "recent_failures": [],
            }

    def _save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_file.with_name(f".{self.state_file.name}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.state, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.state_file)

    def _offset_for(self, path: Path) -> int:
        stat = path.stat()
        key = str(path)
        tracked = self.state["files"].get(key)
        if tracked is None:
            offset = stat.st_size if self.start_at_end and not self.state["initialized"] else 0
            self.state["files"][key] = {"inode": stat.st_ino, "offset": offset}
            return offset
        if tracked.get("inode") != stat.st_ino or stat.st_size < tracked.get("offset", 0):
            tracked["inode"] = stat.st_ino
            tracked["offset"] = 0
        return int(tracked["offset"])

    def _set_offset(self, path: Path, offset: int) -> None:
        self.state["files"][str(path)]["offset"] = offset

    def _should_alert(self) -> bool:
        if self.state["alerted_in_current_streak"]:
            return False
        if self.state["consecutive_failures"] < self.threshold:
            return False
        if self.require_both_tools:
            counts = self.state["failure_counts"]
            return counts.get("search", 0) > 0 and counts.get("visit", 0) > 0
        return True

    def _process_event(self, path: Path, line: str) -> None:
        match = TERMINAL_EVENT_RE.search(line)
        if match is None:
            return
        outcome = match.group("outcome")
        tool = match.group("tool")
        if outcome == "success":
            if self.state["consecutive_failures"]:
                self.logger.info(
                    "Tool success reset failure streak: tool=%s previous_streak=%d",
                    tool,
                    self.state["consecutive_failures"],
                )
            self.state["consecutive_failures"] = 0
            self.state["failure_counts"] = {"search": 0, "visit": 0}
            self.state["alerted_in_current_streak"] = False
            self.state["recent_failures"] = []
            return

        self.state["consecutive_failures"] += 1
        self.state["failure_counts"][tool] = self.state["failure_counts"].get(tool, 0) + 1
        recent = self.state["recent_failures"]
        recent.append({"tool": tool, "log_file": path.name, "line": line.strip()[-600:]})
        del recent[:-20]
        self.logger.warning(
            "Terminal tool failure: tool=%s streak=%d threshold=%d",
            tool,
            self.state["consecutive_failures"],
            self.threshold,
        )
        if self._should_alert():
            payload = {
                "consecutive_failures": self.state["consecutive_failures"],
                "failure_counts": dict(self.state["failure_counts"]),
                "log_dir": str(self.log_dir),
                "recent_failures": list(recent),
            }
            if self.alert_callback(payload):
                self.state["alerted_in_current_streak"] = True
                self.logger.warning(
                    "Tool-failure AutoDL alert sent: streak=%d", payload["consecutive_failures"]
                )

    def scan_once(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.log_dir.glob("*.log")):
            try:
                offset = self._offset_for(path)
                with path.open("rb") as handle:
                    handle.seek(offset)
                    data = handle.read()
            except FileNotFoundError:
                continue
            except OSError as error:
                self.logger.warning("Cannot read %s: %s", path, error)
                continue

            consumed = 0
            for segment in data.splitlines(keepends=True):
                if not segment.endswith((b"\n", b"\r")):
                    break
                consumed += len(segment)
                self._process_event(path, segment.decode("utf-8", errors="replace"))
            if consumed:
                self._set_offset(path, offset + consumed)
        self.state["initialized"] = True
        self._save_state()


def make_autodl_alert(
    token: str, *, name: str, dry_run: bool, logger: logging.Logger
) -> Callable[[dict], bool]:
    def alert(payload: dict) -> bool:
        title = "OpenSeeker RL 工具连续失败告警"
        content = "\n".join(
            [
                "RL 工具环境可能异常，请尽快检查或暂停训练。",
                f"连续终态失败次数：{payload['consecutive_failures']}",
                f"失败分布：{json.dumps(payload['failure_counts'], ensure_ascii=False)}",
                f"日志目录：{payload['log_dir']}",
                "最近失败：",
                *( 
                    f"- {event['tool']} / {event['log_file']}: {event['line']}"
                    for event in payload["recent_failures"]
                ),
            ]
        )
        message = {"title": title, "name": name, "content": content}
        if dry_run:
            logger.warning("Dry-run AutoDL WeChat alert: %s", json.dumps(message, ensure_ascii=False))
            return True
        try:
            response = requests.post(
                AUTODL_WECHAT_URL,
                json=message,
                headers={"Authorization": token},
                timeout=30,
            )
            response.raise_for_status()
            logger.info("AutoDL WeChat alert accepted: status=%d", response.status_code)
            return True
        except requests.RequestException as error:
            logger.error("Failed to send AutoDL WeChat alert: %s", error)
            return False

    return alert


def configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("openseeker.tool_failure_monitor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        default=os.getenv("OPENSEEKER_AGENT_LOG_DIR"),
        help="RL agent_logs directory (default: OPENSEEKER_AGENT_LOG_DIR)",
    )
    parser.add_argument("--threshold", type=int, default=int(os.getenv("OPENSEEKER_TOOL_FAILURE_THRESHOLD", "10")))
    parser.add_argument("--poll-seconds", type=float, default=float(os.getenv("OPENSEEKER_TOOL_FAILURE_POLL_SECONDS", "10")))
    parser.add_argument(
        "--require-both-tools",
        action="store_true",
        help="Alert only if the current failure streak includes both search and visit.",
    )
    parser.add_argument(
        "--start-at",
        choices=("end", "beginning"),
        default="end",
        help="On first run, ignore existing lines (end) or inspect all logs (beginning).",
    )
    parser.add_argument("--state-file")
    parser.add_argument("--monitor-log")
    parser.add_argument(
        "--autodl-token", default=os.getenv("OPENSEEKER_AUTODL_TOKEN", ""),
        help="AutoDL Token for the WeChat message API (default: OPENSEEKER_AUTODL_TOKEN).",
    )
    parser.add_argument(
        "--autodl-name", default=os.getenv("OPENSEEKER_AUTODL_MESSAGE_NAME", "OpenSeeker RL"),
        help="AutoDL message sender name.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Log alert content instead of sending a WeChat message.")
    return parser.parse_args()


def main() -> None:
    secrets_file = Path(
        os.getenv("OPENSEEKER_TOOL_SECRETS_FILE", str(DEFAULT_SECRETS_FILE))
    )
    try:
        load_autodl_secret_from_file(secrets_file)
    except (OSError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    args = parse_args()
    if not args.log_dir:
        raise SystemExit("--log-dir or OPENSEEKER_AGENT_LOG_DIR is required")
    if args.threshold < 1:
        raise SystemExit("--threshold must be at least 1")
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")
    if not args.autodl_token and not args.dry_run:
        raise SystemExit("--autodl-token or OPENSEEKER_AUTODL_TOKEN is required")

    log_dir = Path(args.log_dir).resolve()
    monitor_dir = log_dir.parent / "monitor"
    state_file = Path(args.state_file).resolve() if args.state_file else monitor_dir / "tool_failure_monitor_state.json"
    monitor_log = Path(args.monitor_log).resolve() if args.monitor_log else monitor_dir / "tool_failure_monitor.log"
    logger = configure_logger(monitor_log)
    monitor = ToolFailureMonitor(
        log_dir=log_dir,
        state_file=state_file,
        threshold=args.threshold,
        require_both_tools=args.require_both_tools,
        start_at_end=args.start_at == "end",
        alert_callback=make_autodl_alert(
            args.autodl_token, name=args.autodl_name, dry_run=args.dry_run, logger=logger
        ),
        logger=logger,
    )
    logger.info(
        "Monitoring %s: threshold=%d require_both_tools=%s start_at=%s",
        log_dir, args.threshold, args.require_both_tools, args.start_at,
    )
    try:
        while True:
            monitor.scan_once()
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        logger.info("Tool-failure monitor stopped")


if __name__ == "__main__":
    main()
