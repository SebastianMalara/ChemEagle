from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from review_artifacts import ArtifactStore, create_artifact_store_from_config


RUNTIME_LOG_ROOT = Path("./data/runtime_logs").resolve()
ACTIVE_RUN_LOG_ROOT = RUNTIME_LOG_ROOT / "active"
APP_LOG_PATH = RUNTIME_LOG_ROOT / "app.jsonl"
LEVEL_ORDER = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_APP_LOGGER_NAME = "review_runtime"
_APP_LOGGER_CONFIGURED = False
_RUN_LOG_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar("review_run_log_context", default={})
_TRACEBACK_FINAL_RE = re.compile(r"^[A-Za-z_][\w.]*:")

_KNOWN_LIBRARY_NOISE_RULES = (
    (
        "torchvision_pretrained_deprecation",
        (
            "torchvision/models/_utils.py",
            "parameter 'pretrained' is deprecated",
        ),
    ),
    (
        "torchvision_weights_deprecation",
        (
            "torchvision/models/_utils.py",
            "arguments other than a weight enum or `none` for 'weights' are deprecated",
        ),
    ),
    (
        "torch_meshgrid_indexing",
        (
            "torch.meshgrid",
            "indexing argument",
        ),
    ),
    (
        "torch_checkpoint_use_reentrant",
        (
            "torch.utils.checkpoint",
            "use_reentrant",
        ),
    ),
    (
        "torch_checkpoint_requires_grad",
        (
            "none of the inputs have requires_grad=true",
        ),
    ),
    (
        "transformers_return_dict_hidden_states",
        (
            "return_dict_in_generate",
            "output_hidden_states",
            "is ignored",
        ),
    ),
    (
        "transformers_encoder_config_dump",
        (
            "config of the encoder:",
        ),
    ),
    (
        "transformers_decoder_config_dump",
        (
            "config of the decoder:",
        ),
    ),
    (
        "huggingface_tokenizers_fork_warning",
        (
            "huggingface/tokenizers: the current process just got forked",
        ),
    ),
)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_app_logging() -> logging.Logger:
    global _APP_LOGGER_CONFIGURED
    logger = logging.getLogger(_APP_LOGGER_NAME)
    if _APP_LOGGER_CONFIGURED:
        return logger
    RUNTIME_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = RotatingFileHandler(APP_LOG_PATH, maxBytes=25 * 1024 * 1024, backupCount=5, encoding="utf-8")
    handler.setFormatter(_AppJsonFormatter())
    logger.addHandler(handler)
    _APP_LOGGER_CONFIGURED = True
    return logger


@contextmanager
def bind_log_context(**context: Any) -> Iterator[None]:
    merged = dict(_RUN_LOG_CONTEXT.get())
    merged.update({key: value for key, value in context.items() if value not in (None, "")})
    token = _RUN_LOG_CONTEXT.set(merged)
    try:
        yield
    finally:
        _RUN_LOG_CONTEXT.reset(token)


class _AppJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": utcnow_iso(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(payload, ensure_ascii=False)


class _RunJsonlHandler(logging.Handler):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._seq = 0
        self._last_mono: Optional[float] = None

    def emit(self, record: logging.LogRecord) -> None:
        now_mono = time.perf_counter()
        with self._lock:
            self._seq += 1
            delta_ms = None if self._last_mono is None else int((now_mono - self._last_mono) * 1000)
            self._last_mono = now_mono
            event_extra = getattr(record, "event_extra", {}) or {}
            event = {
                "seq": self._seq,
                "ts": utcnow_iso(),
                "level": record.levelname,
                "logger": record.name,
                "event": getattr(record, "event_name", "") or "log",
                "message": record.getMessage(),
                "run_id": getattr(record, "run_id", ""),
                "experiment_id": getattr(record, "experiment_id", ""),
                "run_source_id": getattr(record, "run_source_id", ""),
                "derived_image_id": getattr(record, "derived_image_id", ""),
                "phase": getattr(record, "phase", ""),
                "provider": getattr(record, "provider", ""),
                "model": getattr(record, "model", ""),
                "delta_ms": delta_ms,
                "duration_ms": getattr(record, "duration_ms", None),
                "thread_name": record.threadName,
                "pid": record.process,
                "exception": logging.Formatter().formatException(record.exc_info) if record.exc_info else "",
                "stream": event_extra.get("stream", ""),
                "category": event_extra.get("category", ""),
                "suppressed": bool(event_extra.get("suppressed", False)),
                "repeat_count": int(event_extra.get("repeat_count", 1) or 1),
                "extra": event_extra,
            }
            self._handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._handle.flush()

    def close(self) -> None:
        try:
            self._handle.close()
        finally:
            super().close()


class _CapturedStream:
    def __init__(self, logger: "BoundRunLogger", path: Path, *, level: str, event_name: str):
        self.logger = logger
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8")
        self._buffer = ""
        self._level = level.upper()
        self._event_name = event_name
        self._pending_kind = ""
        self._pending_lines: List[str] = []

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._handle.write(text)
        self._handle.flush()
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._consume_line(line.rstrip())
        return len(text)

    def flush(self) -> None:
        self._handle.flush()
        trailing = self._buffer.rstrip()
        if trailing:
            self._consume_line(trailing)
        self._buffer = ""
        self._emit_pending()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._handle.close()

    def isatty(self) -> bool:
        return False

    def _consume_line(self, line: str) -> None:
        text = line.rstrip()
        if not text:
            return
        if self._pending_kind:
            if self._should_continue_pending(text):
                self._pending_lines.append(text)
                if self._pending_kind == "config_dump" and text.strip() == "}":
                    self._emit_pending()
                elif self._pending_kind == "traceback" and _is_traceback_terminal_line(text):
                    self._emit_pending()
                return
            self._emit_pending()
        kind = _detect_block_kind(text)
        if kind:
            self._pending_kind = kind
            self._pending_lines = [text]
            if kind == "warning" and not _warning_block_requires_continuation(text):
                self._emit_pending()
            return
        self._emit_text(text)

    def _should_continue_pending(self, text: str) -> bool:
        stripped = text.strip()
        if self._pending_kind == "warning":
            return stripped.startswith("warnings.warn")
        if self._pending_kind == "config_dump":
            return True
        if self._pending_kind == "traceback":
            return (
                text.startswith(" ")
                or stripped.startswith("File ")
                or stripped.startswith("Traceback ")
                or stripped.startswith("The above exception")
                or stripped.startswith("During handling")
                or _is_traceback_terminal_line(text)
            )
        return False

    def _emit_pending(self) -> None:
        if not self._pending_lines:
            self._pending_kind = ""
            return
        self._emit_text("\n".join(self._pending_lines))
        self._pending_kind = ""
        self._pending_lines = []

    def _emit_text(self, text: str) -> None:
        classification = classify_stream_text(text, self._event_name, default_level=self._level)
        self.logger.log(
            classification["level"],
            self._event_name,
            text,
            stream=self._event_name,
            category=classification["category"],
            suppressed=classification["suppressed"],
            signature=classification["signature"],
        )


class _ConsoleMirrorHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        extra = getattr(record, "event_extra", {}) or {}
        if extra.get("suppressed"):
            return
        stream = getattr(sys, "__stderr__", None) or sys.stderr
        if stream is None:
            return
        line = _format_console_record(record)
        with self._lock:
            stream.write(line + "\n")
            stream.flush()


class BoundRunLogger:
    def __init__(self, logger: logging.Logger, base_context: Dict[str, Any]):
        self._logger = logger
        self._base_context = {key: value for key, value in base_context.items() if value not in (None, "")}

    def bind(self, **context: Any) -> "BoundRunLogger":
        merged = dict(self._base_context)
        merged.update({key: value for key, value in context.items() if value not in (None, "")})
        return BoundRunLogger(self._logger, merged)

    def log(self, level: str, event_name: str, message: str, **extra: Any) -> None:
        context = dict(_RUN_LOG_CONTEXT.get())
        context.update(self._base_context)
        payload = {
            "event_name": event_name,
            "event_extra": {key: value for key, value in extra.items() if value not in (None, "")},
        }
        payload.update(context)
        self._logger.log(LEVEL_ORDER.get(level.upper(), logging.INFO), message, extra=payload)
        configure_app_logging().log(
            LEVEL_ORDER.get(level.upper(), logging.INFO),
            f"{event_name}: {message}",
        )

    def debug(self, event_name: str, message: str, **extra: Any) -> None:
        self.log("DEBUG", event_name, message, **extra)

    def info(self, event_name: str, message: str, **extra: Any) -> None:
        self.log("INFO", event_name, message, **extra)

    def warning(self, event_name: str, message: str, **extra: Any) -> None:
        self.log("WARNING", event_name, message, **extra)

    def error(self, event_name: str, message: str, **extra: Any) -> None:
        self.log("ERROR", event_name, message, **extra)

    def exception(self, event_name: str, message: str, **extra: Any) -> None:
        context = dict(_RUN_LOG_CONTEXT.get())
        context.update(self._base_context)
        payload = {
            "event_name": event_name,
            "event_extra": {key: value for key, value in extra.items() if value not in (None, "")},
        }
        payload.update(context)
        self._logger.exception(message, extra=payload)
        configure_app_logging().exception(f"{event_name}: {message}")


class RunLogSession:
    def __init__(self, *, run_id: str, experiment_id: str = ""):
        configure_app_logging()
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.active_dir = ACTIVE_RUN_LOG_ROOT / run_id
        self.active_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.active_dir / "events.jsonl"
        self.stdout_path = self.active_dir / "stdout.log"
        self.stderr_path = self.active_dir / "stderr.log"

        self._logger = logging.getLogger(f"review_runtime.run.{run_id}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        self._handler = _RunJsonlHandler(self.events_path)
        self._console_handler = _ConsoleMirrorHandler()
        self._logger.handlers = [self._handler, self._console_handler]
        self.logger = BoundRunLogger(
            self._logger,
            {
                "run_id": run_id,
                "experiment_id": experiment_id,
            },
        )

    @contextmanager
    def capture_streams(self, logger: Optional[BoundRunLogger] = None) -> Iterator[None]:
        active_logger = logger or self.logger
        stdout_stream = _CapturedStream(active_logger, self.stdout_path, level="INFO", event_name="stdout")
        stderr_stream = _CapturedStream(active_logger, self.stderr_path, level="ERROR", event_name="stderr")
        try:
            with redirect_stdout(stdout_stream), redirect_stderr(stderr_stream):
                yield
        finally:
            stdout_stream.close()
            stderr_stream.close()

    def finalize(self, store: ArtifactStore) -> Dict[str, str]:
        self._handler.flush()
        events_key = f"logs/{self.run_id}/events.jsonl"
        stdout_key = f"logs/{self.run_id}/stdout.log"
        stderr_key = f"logs/{self.run_id}/stderr.log"
        store.put_file(events_key, str(self.events_path), content_type="application/json")
        store.put_file(stdout_key, str(self.stdout_path), content_type="text/plain")
        store.put_file(stderr_key, str(self.stderr_path), content_type="text/plain")
        return {
            "log_artifact_key": events_key,
            "stdout_artifact_key": stdout_key,
            "stderr_artifact_key": stderr_key,
        }

    def close(self) -> None:
        if self._handler in self._logger.handlers:
            self._logger.removeHandler(self._handler)
        if self._console_handler in self._logger.handlers:
            self._logger.removeHandler(self._console_handler)
        self._handler.close()
        self._console_handler.close()


def read_log_tail(
    *,
    run_id: str,
    config: Dict[str, Any],
    log_artifact_key: str,
    tail_lines: int = 200,
    min_level: str = "INFO",
    raw: bool = False,
    include_suppressed: bool = False,
) -> Dict[str, Any]:
    active_path = ACTIVE_RUN_LOG_ROOT / run_id / "events.jsonl"
    if active_path.exists():
        lines = _tail_lines(active_path, max(tail_lines * 10, tail_lines))
    elif log_artifact_key:
        store = create_artifact_store_from_config(config)
        lines = store.get_bytes(log_artifact_key).decode("utf-8", errors="replace").splitlines()[-max(tail_lines * 10, tail_lines):]
    else:
        lines = []
    events = _parse_events(lines, min_level=min_level, limit=tail_lines, include_suppressed=include_suppressed)
    return {
        "raw": "\n".join(json.dumps(event, ensure_ascii=False) for event in events) if raw else "",
        "formatted": "\n".join(_format_event(event) for event in events),
        "events": events,
    }


def get_log_download_ref(*, run_id: str, config: Dict[str, Any], log_artifact_key: str) -> str:
    active_path = ACTIVE_RUN_LOG_ROOT / run_id / "events.jsonl"
    if active_path.exists():
        return str(active_path)
    if not log_artifact_key:
        return ""
    store = create_artifact_store_from_config(config)
    return store.get_download_ref(log_artifact_key)


def _tail_lines(path: Path, count: int) -> List[str]:
    buffer: deque[str] = deque(maxlen=max(1, count))
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            buffer.append(line.rstrip("\n"))
    return list(buffer)


def _parse_events(lines: Iterable[str], *, min_level: str, limit: int, include_suppressed: bool) -> List[Dict[str, Any]]:
    threshold = LEVEL_ORDER.get((min_level or "INFO").upper(), logging.INFO)
    events: List[Dict[str, Any]] = []
    dedup_index: Dict[str, int] = {}
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload.setdefault("stream", payload.get("extra", {}).get("stream", ""))
        payload.setdefault("category", payload.get("extra", {}).get("category", ""))
        payload["suppressed"] = bool(payload.get("suppressed", payload.get("extra", {}).get("suppressed", False)))
        payload["repeat_count"] = int(payload.get("repeat_count", payload.get("extra", {}).get("repeat_count", 1) or 1))
        if LEVEL_ORDER.get(str(payload.get("level") or "INFO").upper(), logging.INFO) < threshold:
            continue
        if payload.get("suppressed") and not include_suppressed:
            continue
        signature = str(payload.get("extra", {}).get("signature") or "")
        if signature and payload.get("level") != "ERROR":
            existing_index = dedup_index.get(signature)
            if existing_index is not None:
                events[existing_index]["repeat_count"] = int(events[existing_index].get("repeat_count", 1)) + int(
                    payload.get("repeat_count", 1)
                )
                continue
            dedup_index[signature] = len(events)
        events.append(payload)
    return events[-limit:]


def _format_event(event: Dict[str, Any]) -> str:
    ts = str(event.get("ts") or "")
    level = str(event.get("level") or "INFO")
    phase = str(event.get("phase") or "")
    prefix = f"[{ts}] {level}"
    if phase:
        prefix += f" {phase}"
    message = str(event.get("message") or "")
    event_name = str(event.get("event") or "")
    if event_name:
        prefix += f" {event_name}"
    extra = event.get("extra", {}) or {}
    details = _format_extra_summary(extra)
    line = f"{prefix} {message}".strip()
    repeat_count = int(event.get("repeat_count", 1) or 1)
    if repeat_count > 1:
        line += f" [repeated x{repeat_count}]"
    if details:
        line += f" [{details}]"
    traceback_text = str(extra.get("traceback") or event.get("exception") or "").strip()
    if traceback_text:
        line += f"\n{traceback_text}"
    return line


def _format_console_record(record: logging.LogRecord) -> str:
    ts = utcnow_iso()
    level = record.levelname
    run_id = getattr(record, "run_id", "")
    phase = getattr(record, "phase", "")
    event_name = getattr(record, "event_name", "") or "log"
    message = record.getMessage()
    prefix = f"[{ts}] {level}"
    if run_id:
        prefix += f" run={run_id}"
    if phase:
        prefix += f" phase={phase}"
    prefix += f" {event_name}"
    extra = getattr(record, "event_extra", {}) or {}
    details = _format_extra_summary(extra)
    line = f"{prefix} {message}".strip()
    if details:
        line += f" [{details}]"
    traceback_text = str(extra.get("traceback") or "").strip()
    if traceback_text:
        line += f"\n{traceback_text}"
    return line


def _format_extra_summary(extra: Dict[str, Any]) -> str:
    details: List[str] = []
    for key in (
        "category",
        "failure_kind",
        "llm_stage",
        "llm_phase",
        "provider",
        "model",
        "base_url",
        "exception_class",
        "cause_class",
        "cause_message",
    ):
        value = extra.get(key)
        if value in (None, ""):
            continue
        details.append(f"{key}={value}")
    return " ".join(details)


def _infer_stream_level(event_name: str, text: str, default_level: str) -> str:
    lowered = (text or "").strip().lower()
    if not lowered:
        return default_level
    if "traceback" in lowered or lowered.startswith("error:") or lowered.startswith("exception:"):
        return "ERROR"
    if "userwarning:" in lowered or lowered.startswith("warning:") or lowered.startswith("[warning]"):
        return "WARNING"
    if event_name == "stderr" and "warning" in lowered and "error" not in lowered:
        return "WARNING"
    return default_level


def is_known_library_noise(text: str) -> str:
    lowered = (text or "").strip().lower()
    if not lowered:
        return ""
    for signature, required_parts in _KNOWN_LIBRARY_NOISE_RULES:
        if all(part in lowered for part in required_parts):
            return signature
    return ""


def classify_stream_text(text: str, stream: str, *, default_level: str = "INFO") -> Dict[str, Any]:
    lowered = (text or "").strip().lower()
    signature = ""
    category = "stream"
    stream_default_level = "INFO" if stream in {"stdout", "stderr"} else default_level
    level = stream_default_level
    suppressed = False

    if _detect_block_kind(text) == "traceback" or "traceback (most recent call last):" in lowered:
        return {
            "level": "ERROR",
            "category": "traceback",
            "suppressed": False,
            "signature": "",
        }

    noise_signature = is_known_library_noise(text)
    if noise_signature:
        category = "library_noise"
        suppressed = True
        signature = noise_signature
        level = "WARNING" if "warning" in lowered else "INFO"
    elif "observer returned unsupported tool" in lowered:
        category = "observer_warning"
        level = "WARNING"
        signature = "observer_unsupported_tool"
    elif "userwarning:" in lowered or lowered.startswith("warning:") or lowered.startswith("[warning]"):
        category = "warning"
        level = "WARNING"
        signature = _normalized_signature(text)
    elif "error:" in lowered or "exception:" in lowered:
        category = "application_error"
        level = "ERROR"
    elif stream == "stderr" and "warning" in lowered and "error" not in lowered:
        category = "warning"
        level = "WARNING"
        signature = _normalized_signature(text)
    else:
        category = "progress" if stream == "stdout" else "stream"
        level = _infer_stream_level(stream, text, stream_default_level)

    return {
        "level": level,
        "category": category,
        "suppressed": suppressed,
        "signature": signature,
    }


def coalesce_stream_block(lines: Iterable[str], stream: str, *, default_level: str = "INFO") -> Dict[str, Any]:
    text = "\n".join(str(line).rstrip() for line in lines if str(line).rstrip())
    classification = classify_stream_text(text, stream, default_level=default_level)
    classification["message"] = text
    return classification


def _detect_block_kind(text: str) -> str:
    stripped = (text or "").strip()
    lowered = stripped.lower()
    if stripped.startswith("Traceback (most recent call last):"):
        return "traceback"
    if stripped.startswith("Config of the encoder:") or stripped.startswith("Config of the decoder:"):
        return "config_dump"
    if "userwarning:" in lowered or lowered.startswith("warning:") or lowered.startswith("[warning]"):
        return "warning"
    return ""


def _warning_block_requires_continuation(text: str) -> bool:
    lowered = (text or "").lower()
    return "userwarning:" in lowered or lowered.startswith("warning:")


def _is_traceback_terminal_line(text: str) -> bool:
    stripped = (text or "").strip()
    return bool(_TRACEBACK_FINAL_RE.match(stripped))


def _normalized_signature(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())
