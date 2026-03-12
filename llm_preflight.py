from __future__ import annotations

import base64
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from llm_profiles import OPENAI_COMPATIBLE_PROVIDERS, LLMProfile, resolve_llm_profile
from runtime_device import resolve_ocr_backend

try:
    from anthropic import Anthropic
    from anthropic import (
        APIConnectionError as AnthropicAPIConnectionError,
        APITimeoutError as AnthropicAPITimeoutError,
        AuthenticationError as AnthropicAuthenticationError,
        BadRequestError as AnthropicBadRequestError,
        RateLimitError as AnthropicRateLimitError,
    )
except ImportError:  # optional dependency
    Anthropic = None
    AnthropicAPIConnectionError = tuple()  # type: ignore[assignment]
    AnthropicAPITimeoutError = tuple()  # type: ignore[assignment]
    AnthropicAuthenticationError = tuple()  # type: ignore[assignment]
    AnthropicBadRequestError = tuple()  # type: ignore[assignment]
    AnthropicRateLimitError = tuple()  # type: ignore[assignment]

try:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        AzureOpenAI,
        BadRequestError,
        InternalServerError,
        NotFoundError,
        OpenAI,
        PermissionDeniedError,
        RateLimitError,
    )
except ImportError:  # pragma: no cover - dependency guard
    APIConnectionError = tuple()  # type: ignore[assignment]
    APITimeoutError = tuple()  # type: ignore[assignment]
    AuthenticationError = tuple()  # type: ignore[assignment]
    AzureOpenAI = None  # type: ignore[assignment]
    BadRequestError = tuple()  # type: ignore[assignment]
    InternalServerError = tuple()  # type: ignore[assignment]
    NotFoundError = tuple()  # type: ignore[assignment]
    OpenAI = None  # type: ignore[assignment]
    PermissionDeniedError = tuple()  # type: ignore[assignment]
    RateLimitError = tuple()  # type: ignore[assignment]


_TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAI0lEQVR4nGP8//8/AymAiSTVDKMaiANMRKqDg1ENxACSNQAAVW0DHUvDbUgAAAAASUVORK5CYII="
)


@dataclass(frozen=True)
class ProviderFailureClass:
    kind: str
    systemic: bool
    retryable: bool
    message: str
    http_status: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderProbeResult:
    ok: bool
    purpose: str
    provider: str
    model: str
    failure_kind: str
    message: str
    http_status: Optional[int]
    supports_images: Optional[bool]
    diagnostic_payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunFailureControllerState:
    systemic_failure_kind: str = ""
    systemic_failure_count: int = 0
    last_systemic_error: str = ""
    abort_reason: str = ""


class RunFailureController:
    def __init__(self, threshold: int = 2):
        self.threshold = threshold
        self.state = RunFailureControllerState()

    def record(
        self,
        failure: ProviderFailureClass,
        *,
        source_index: int,
        source_name: str,
    ) -> Tuple[bool, str]:
        if not failure.systemic:
            self.state.systemic_failure_kind = ""
            self.state.systemic_failure_count = 0
            self.state.last_systemic_error = ""
            self.state.abort_reason = ""
            return False, ""

        same_kind = failure.kind == self.state.systemic_failure_kind
        self.state.systemic_failure_kind = failure.kind
        self.state.systemic_failure_count = self.state.systemic_failure_count + 1 if same_kind else 1
        self.state.last_systemic_error = failure.message

        immediate_first_source = failure.kind in {
            "auth_error",
            "unsupported_model_or_capability",
        } or (source_index == 0 and failure.kind in {"endpoint_unreachable", "dns_or_connection_error"})

        repeated_abort = self.state.systemic_failure_count >= self.threshold
        if not immediate_first_source and not repeated_abort:
            return False, ""

        reason = (
            f"Aborting batch on systemic provider failure '{failure.kind}' while processing {source_name}: "
            f"{failure.message}"
        )
        self.state.abort_reason = reason
        return True, reason


def classify_provider_exception(exc: Exception) -> ProviderFailureClass:
    message = str(exc).strip() or exc.__class__.__name__
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    lowered = message.lower()

    if isinstance(exc, (AuthenticationError, PermissionDeniedError, AnthropicAuthenticationError)):
        return ProviderFailureClass("auth_error", True, False, message, status_code)
    if isinstance(exc, (APIConnectionError, AnthropicAPIConnectionError)):
        kind = "endpoint_unreachable" if any(token in lowered for token in ["connection refused", "refused"]) else "dns_or_connection_error"
        return ProviderFailureClass(kind, True, False, message, status_code)
    if isinstance(exc, (APITimeoutError, AnthropicAPITimeoutError)):
        return ProviderFailureClass("timeout", True, True, message, status_code)
    if isinstance(exc, (RateLimitError, AnthropicRateLimitError)):
        return ProviderFailureClass("rate_limited", False, True, message, status_code)
    if isinstance(exc, InternalServerError):
        return ProviderFailureClass("provider_overloaded", True, True, message, status_code)
    if isinstance(exc, (BadRequestError, AnthropicBadRequestError, NotFoundError)):
        if any(token in lowered for token in ["unsupported parameter", "max_tokens", "max_completion_tokens", "response_format", "temperature"]):
            return ProviderFailureClass("bad_request_non_retryable", False, False, message, status_code)
        if "image_parse_error" in lowered or "please make sure your image is valid" in lowered:
            return ProviderFailureClass("bad_request_non_retryable", False, False, message, status_code)
        if any(token in lowered for token in ["does not support image", "vision", "multimodal"]):
            return ProviderFailureClass("unsupported_model_or_capability", True, False, message, status_code)
        if any(token in lowered for token in ["model", "deployment", "not found", "unknown model"]):
            return ProviderFailureClass("unsupported_model_or_capability", True, False, message, status_code)
        return ProviderFailureClass("bad_request_non_retryable", False, False, message, status_code)

    if any(token in lowered for token in ["connection error", "connection reset", "temporary failure in name resolution", "name or service not known"]):
        return ProviderFailureClass("dns_or_connection_error", True, False, message, status_code)
    if any(token in lowered for token in ["timed out", "timeout"]):
        return ProviderFailureClass("timeout", True, True, message, status_code)
    if any(token in lowered for token in ["rate limit", "429"]):
        return ProviderFailureClass("rate_limited", False, True, message, status_code)
    if any(token in lowered for token in ["unsupported parameter", "max_tokens", "max_completion_tokens", "image_parse_error"]):
        return ProviderFailureClass("bad_request_non_retryable", False, False, message, status_code)
    if any(token in lowered for token in ["unauthorized", "invalid api key", "authentication", "forbidden"]):
        return ProviderFailureClass("auth_error", True, False, message, status_code)
    if any(token in lowered for token in ["model", "deployment", "vision", "multimodal"]) and any(
        token in lowered for token in ["unsupported", "not found", "invalid", "does not support"]
    ):
        return ProviderFailureClass("unsupported_model_or_capability", True, False, message, status_code)
    if any(token in lowered for token in ["overloaded", "service unavailable", "502", "503", "504"]):
        return ProviderFailureClass("provider_overloaded", True, True, message, status_code)
    return ProviderFailureClass("unknown_provider_error", False, False, message, status_code)


def profile_probe_key(profile: LLMProfile, *, purpose: str) -> str:
    return "|".join(
        [
            purpose,
            profile.scope,
            profile.provider,
            profile.model,
            profile.base_url,
            profile.azure_endpoint,
            profile.api_version,
            "key-present" if bool(profile.api_key) else "key-missing",
        ]
    )


def collect_runtime_provider_preflight(
    *,
    profile_configs: Iterable[Dict[str, Any]],
    mode: str,
    resolved_ocr_backend: Optional[str] = None,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    blocking_errors: List[str] = []
    warnings: List[str] = []
    seen: set[str] = set()

    for config in profile_configs:
        required_scopes = [("main", "main_text", mode == "cloud")]
        raw_ocr_backend = resolved_ocr_backend or str(config.get("OCR_BACKEND") or config.get("ocr_backend") or "auto")
        ocr_backend = resolve_ocr_backend(raw_ocr_backend, str(config.get("CHEMEAGLE_RUN_MODE") or config.get("mode") or mode))
        if ocr_backend == "llm_vision":
            required_scopes.append(("ocr", "ocr_vision", True))
        for scope, purpose, required in required_scopes:
            if not required:
                continue
            profile = resolve_llm_profile(scope=scope, values=config, default_model="gpt-5-mini")
            key = profile_probe_key(profile, purpose=purpose)
            if key in seen:
                continue
            seen.add(key)
            probe = probe_llm_profile(profile, purpose=purpose)
            probe_payload = probe.to_dict()
            probe_payload["required"] = required
            results.append(probe_payload)
            if not probe.ok and required:
                blocking_errors.append(f"{purpose}: {probe.message}")
            elif not probe.ok:
                warnings.append(f"{purpose}: {probe.message}")

    return {
        "results": results,
        "blocking_errors": blocking_errors,
        "warnings": warnings,
        "status": "failed_blocking" if blocking_errors else ("warning" if warnings else "passed"),
    }


def probe_llm_profile(profile: LLMProfile, purpose: str) -> ProviderProbeResult:
    try:
        if purpose == "ocr_vision":
            _perform_vision_probe(profile)
            supports_images: Optional[bool] = True
        else:
            _perform_text_probe(profile)
            supports_images = None
        return ProviderProbeResult(
            ok=True,
            purpose=purpose,
            provider=profile.provider,
            model=profile.model,
            failure_kind="",
            message="Probe passed.",
            http_status=None,
            supports_images=supports_images,
            diagnostic_payload={
                "provider": profile.provider,
                "model": profile.model,
                "base_url": profile.base_url,
                "azure_endpoint": profile.azure_endpoint,
                "api_version": profile.api_version,
            },
        )
    except Exception as exc:
        failure = classify_provider_exception(exc)
        return ProviderProbeResult(
            ok=False,
            purpose=purpose,
            provider=profile.provider,
            model=profile.model,
            failure_kind=failure.kind,
            message=failure.message,
            http_status=failure.http_status,
            supports_images=False if purpose == "ocr_vision" else None,
            diagnostic_payload={
                "provider": profile.provider,
                "model": profile.model,
                "base_url": profile.base_url,
                "azure_endpoint": profile.azure_endpoint,
                "api_version": profile.api_version,
                "failure": failure.to_dict(),
            },
        )


def _perform_text_probe(profile: LLMProfile) -> None:
    if profile.provider == "azure":
        if AzureOpenAI is None:
            raise ImportError("openai package is required for Azure probes.")
        client = AzureOpenAI(
            api_key=profile.api_key,
            api_version=profile.api_version or "2024-06-01",
            azure_endpoint=profile.azure_endpoint,
            timeout=10.0,
        )
        _create_openai_probe_completion(
            client=client,
            model=profile.model,
            messages=[{"role": "user", "content": "Reply with OK."}],
        )
        return

    if profile.provider in OPENAI_COMPATIBLE_PROVIDERS:
        if OpenAI is None:
            raise ImportError("openai package is required for provider probes.")
        kwargs: Dict[str, Any] = {"api_key": profile.api_key or "EMPTY", "timeout": 10.0}
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        elif profile.provider == "openai":
            kwargs["base_url"] = "https://api.openai.com/v1"
        client = OpenAI(**kwargs)
        _create_openai_probe_completion(
            client=client,
            model=profile.model,
            messages=[{"role": "user", "content": "Reply with OK."}],
        )
        return

    if profile.provider == "anthropic":
        if Anthropic is None:
            raise ImportError("anthropic package is not installed.")
        client = Anthropic(api_key=profile.api_key, timeout=10.0)
        client.messages.create(
            model=profile.model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=5,
        )
        return

    raise ValueError(f"Unsupported provider for runtime probe: {profile.provider}")


def _perform_vision_probe(profile: LLMProfile) -> None:
    data_url = f"data:image/png;base64,{_TINY_PNG_BASE64}"
    if profile.provider == "azure":
        if AzureOpenAI is None:
            raise ImportError("openai package is required for Azure probes.")
        client = AzureOpenAI(
            api_key=profile.api_key,
            api_version=profile.api_version or "2024-06-01",
            azure_endpoint=profile.azure_endpoint,
            timeout=10.0,
        )
        _create_openai_probe_completion(
            client=client,
            model=profile.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reply with OK."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        return

    if profile.provider in OPENAI_COMPATIBLE_PROVIDERS:
        if OpenAI is None:
            raise ImportError("openai package is required for provider probes.")
        kwargs: Dict[str, Any] = {"api_key": profile.api_key or "EMPTY", "timeout": 10.0}
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        elif profile.provider == "openai":
            kwargs["base_url"] = "https://api.openai.com/v1"
        client = OpenAI(**kwargs)
        _create_openai_probe_completion(
            client=client,
            model=profile.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reply with OK."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        return

    if profile.provider == "anthropic":
        if Anthropic is None:
            raise ImportError("anthropic package is not installed.")
        client = Anthropic(api_key=profile.api_key, timeout=10.0)
        client.messages.create(
            model=profile.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reply with OK."},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": _TINY_PNG_BASE64,
                            },
                        },
                    ],
                }
            ],
            max_tokens=5,
        )
        return

    raise ValueError(f"Unsupported provider for vision probe: {profile.provider}")


def _create_openai_probe_completion(*, client: Any, model: str, messages: List[Dict[str, Any]]) -> Any:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    try:
        return client.chat.completions.create(**kwargs)
    except BadRequestError as exc:
        if not _should_retry_openai_probe_with_completion_budget(exc):
            raise
        return _retry_openai_probe_with_completion_budget(client=client, kwargs=kwargs)
    except Exception as exc:
        if not _looks_like_probe_completion_budget_error(exc):
            raise
        return _retry_openai_probe_with_completion_budget(client=client, kwargs=kwargs)


def _retry_openai_probe_with_completion_budget(*, client: Any, kwargs: Dict[str, Any]) -> Any:
    probe_budget = max(16, int(os.getenv("LLM_PREFLIGHT_OPENAI_MAX_TOKENS", "64")))
    completion_kwargs = dict(kwargs)
    completion_kwargs["max_completion_tokens"] = probe_budget
    try:
        return client.chat.completions.create(**completion_kwargs)
    except BadRequestError as exc:
        if not _is_max_completion_tokens_error(exc):
            raise
        fallback_kwargs = dict(kwargs)
        fallback_kwargs["max_tokens"] = probe_budget
        return client.chat.completions.create(**fallback_kwargs)
    except Exception as exc:
        if not _looks_like_max_completion_tokens_error(exc):
            raise
        fallback_kwargs = dict(kwargs)
        fallback_kwargs["max_tokens"] = probe_budget
        return client.chat.completions.create(**fallback_kwargs)


def _should_retry_openai_probe_with_completion_budget(exc: BadRequestError) -> bool:
    body = getattr(exc, "body", {}) or {}
    error = body.get("error", {}) if isinstance(body, dict) else {}
    param = str(error.get("param") or "").strip().lower()
    code = str(error.get("code") or "").strip().lower()
    message = str(error.get("message") or exc).lower()
    if param in {"max_tokens", "max_completion_tokens"}:
        return True
    if code == "unsupported_parameter" and "max_completion_tokens" in message:
        return False
    if "output limit" in message and "max_tokens" in message:
        return True
    return "higher max_tokens" in message or ("max tokens" in message and "required" in message)


def _looks_like_probe_completion_budget_error(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return "higher max_tokens" in lowered or ("output limit" in lowered and "max_tokens" in lowered)


def _is_max_completion_tokens_error(exc: BadRequestError) -> bool:
    body = getattr(exc, "body", {}) or {}
    error = body.get("error", {}) if isinstance(body, dict) else {}
    param = str(error.get("param") or "").strip().lower()
    code = str(error.get("code") or "").strip().lower()
    message = str(error.get("message") or exc).lower()
    return param == "max_completion_tokens" or code == "unsupported_parameter" and "max_completion_tokens" in message


def _looks_like_max_completion_tokens_error(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return "max_completion_tokens" in lowered and "unsupported" in lowered
