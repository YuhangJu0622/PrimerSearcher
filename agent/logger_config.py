import logging
import os
from contextvars import ContextVar
from typing import Dict, Optional

current_qid: ContextVar[Optional[int]] = ContextVar("current_qid", default=None)

_LOGGER_NAME = "research_agent"
_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_initialized = False


def get_logger() -> logging.Logger:
    """Get the project-wide logger, initializing on first call."""
    global _initialized
    if not _initialized:
        _initialized = True
        logger = logging.getLogger(_LOGGER_NAME)
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))
        logger.propagate = False

        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(console)
    return logging.getLogger(_LOGGER_NAME)


class PerQuestionHandler(logging.Handler):
    """Route log records to per-question files based on contextvars."""

    def __init__(self, log_dir: str):
        super().__init__()
        self.log_dir = log_dir
        self._handlers: Dict[int, logging.FileHandler] = {}

    def _get_file_handler(self, qid: int) -> logging.FileHandler:
        if qid not in self._handlers:
            fh = logging.FileHandler(
                os.path.join(self.log_dir, f"question_{qid}.log"),
                encoding="utf-8",
            )
            fh.setFormatter(self.formatter)
            self._handlers[qid] = fh
        return self._handlers[qid]

    def emit(self, record: logging.LogRecord) -> None:
        qid = current_qid.get()
        if qid is not None:
            self._get_file_handler(qid).emit(record)

    def close(self) -> None:
        for fh in self._handlers.values():
            fh.close()
        self._handlers.clear()
        super().close()


def setup_eval_logging(log_dir: str) -> PerQuestionHandler:
    """Attach a PerQuestionHandler to the project logger for evaluation runs.

    Returns the handler so the caller can close() it when done.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger()
    handler = PerQuestionHandler(log_dir)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    return handler
