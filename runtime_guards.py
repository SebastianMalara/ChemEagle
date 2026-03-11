from __future__ import annotations

import json
from typing import Any, Iterable, Sequence


class RuntimeStageError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        context: str = "",
        retry_trigger: str = "",
    ) -> None:
        full_message = f"{context}: {message}" if context else message
        super().__init__(full_message)
        self.context = context
        self.retry_trigger = retry_trigger
        self._retry_trigger = retry_trigger
        self._runtime_stage = context


def attach_runtime_metadata(exc: Exception, **metadata: Any) -> Exception:
    for key, value in metadata.items():
        if value in (None, "", [], {}):
            continue
        setattr(exc, key, value)
    return exc


def _raise_runtime_error(message: str, *, context: str, retry_trigger: str = "") -> RuntimeStageError:
    raise RuntimeStageError(message, context=context, retry_trigger=retry_trigger)


def first_item(seq: Any, *, context: str = "", default: Any = None) -> Any:
    if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes, bytearray)):
        return seq[0] if seq else default
    return default


def require_first_item(seq: Any, *, context: str, retry_trigger: str = "") -> Any:
    item = first_item(seq, context=context, default=None)
    if item is None:
        _raise_runtime_error("empty result list", context=context, retry_trigger=retry_trigger)
    return item


def first_choice(response: Any, *, context: str, retry_trigger: str = "") -> Any:
    return require_first_item(getattr(response, "choices", None), context=f"{context}: response choices", retry_trigger=retry_trigger)


def assistant_message(response: Any, *, context: str, retry_trigger: str = "") -> Any:
    choice = first_choice(response, context=context, retry_trigger=retry_trigger)
    message = getattr(choice, "message", None)
    if message is None:
        _raise_runtime_error("missing assistant message", context=context, retry_trigger=retry_trigger)
    return message


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def message_content(
    response: Any,
    *,
    context: str,
    default: str = "",
    required: bool = False,
    retry_trigger: str = "",
) -> str:
    message = assistant_message(response, context=context, retry_trigger=retry_trigger)
    text = _content_to_text(getattr(message, "content", None))
    if text:
        return text
    if required:
        _raise_runtime_error("assistant message content is empty", context=context, retry_trigger=retry_trigger)
    return default


def tool_calls_or_empty(response_or_message: Any, *, context: str) -> list[Any]:
    message = response_or_message
    if hasattr(response_or_message, "choices"):
        message = assistant_message(response_or_message, context=context)
    tool_calls = getattr(message, "tool_calls", None)
    return list(tool_calls or [])


def first_tool_call(response_or_message: Any, *, context: str, retry_trigger: str = "") -> Any:
    tool_calls = tool_calls_or_empty(response_or_message, context=context)
    if not tool_calls:
        _raise_runtime_error("assistant message has no tool calls", context=context, retry_trigger=retry_trigger)
    return tool_calls[0]


def first_dict_item(seq: Any, *, context: str, retry_trigger: str = "") -> dict[str, Any]:
    item = require_first_item(seq, context=context, retry_trigger=retry_trigger)
    if not isinstance(item, dict):
        _raise_runtime_error("first result is not a dictionary", context=context, retry_trigger=retry_trigger)
    return item


def nested_first(seq: Any, *, context: str, retry_trigger: str = "") -> Any:
    return require_first_item(seq, context=context, retry_trigger=retry_trigger)


def safe_json_loads(raw: Any, *, context: str, retry_trigger: str = "", default: Any = None) -> Any:
    if raw is None or raw == "":
        if default is not None:
            return default
        _raise_runtime_error("empty JSON content", context=context, retry_trigger=retry_trigger)
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        if default is not None:
            return default
        _raise_runtime_error(f"invalid JSON content: {exc}", context=context, retry_trigger=retry_trigger)
