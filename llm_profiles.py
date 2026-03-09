from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

try:
    from anthropic import Anthropic
except ImportError:  # optional dependency
    Anthropic = None


OPENAI_COMPATIBLE_PROVIDERS = {"openai", "openai_compatible", "lmstudio", "local_openai"}
MANUAL_MODEL_LIST_PROVIDERS = {"azure"}


@dataclass
class LLMProfile:
    scope: str
    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""
    azure_endpoint: str = ""
    api_version: str = ""
    inherit_main: bool = False


@dataclass
class ModelOption:
    id: str
    label: str
    provider: str
    raw: Any = None


@dataclass
class ModelCatalogResult:
    ok: bool
    provider: str
    models: list[ModelOption]
    status: str
    manual_only: bool = False


def env_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _get_value(values: dict[str, Any] | None, *keys: str, default: str = "") -> str:
    if values:
        for key in keys:
            value = values.get(key)
            if value not in (None, ""):
                return str(value)
    for key in keys:
        value = os.getenv(key)
        if value not in (None, ""):
            return value
    return default


def _resolve_main_provider(values: dict[str, Any] | None) -> str:
    return _get_value(values, "LLM_PROVIDER", default="azure").strip().lower() or "azure"


def _resolve_main_model(values: dict[str, Any] | None, default_model: str) -> str:
    return _get_value(values, "LLM_MODEL", default=default_model).strip() or default_model


def _resolve_openai_like_api_key(values: dict[str, Any] | None, *, prefix: str = "") -> str:
    keys = [
        f"{prefix}LLM_API_KEY",
        f"{prefix}OPENAI_API_KEY",
        f"{prefix}VLLM_API_KEY",
        f"{prefix}API_KEY",
        f"{prefix}LMSTUDIO_API_KEY",
        f"{prefix}OLLAMA_API_KEY",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "VLLM_API_KEY",
        "API_KEY",
        "LMSTUDIO_API_KEY",
        "OLLAMA_API_KEY",
    ]
    return _get_value(values, *keys)


def _resolve_openai_like_base_url(values: dict[str, Any] | None, *, prefix: str = "") -> str:
    keys = [
        f"{prefix}LLM_BASE_URL",
        f"{prefix}OPENAI_BASE_URL",
        f"{prefix}VLLM_BASE_URL",
        f"{prefix}LMSTUDIO_BASE_URL",
        f"{prefix}OLLAMA_BASE_URL",
        "LLM_BASE_URL",
        "OPENAI_BASE_URL",
        "VLLM_BASE_URL",
        "LMSTUDIO_BASE_URL",
        "OLLAMA_BASE_URL",
    ]
    return _get_value(values, *keys)


def resolve_llm_profile(
    *,
    scope: str = "main",
    values: dict[str, Any] | None = None,
    default_model: str = "gpt-5-mini",
) -> LLMProfile:
    if scope not in {"main", "ocr"}:
        raise ValueError(f"Unsupported LLM profile scope: {scope}")

    main_provider = _resolve_main_provider(values)
    main_model = _resolve_main_model(values, default_model)

    main_profile = LLMProfile(
        scope="main",
        provider=main_provider,
        model=main_model,
        api_key="",
        base_url="",
        azure_endpoint="",
        api_version=_get_value(values, "API_VERSION", "AZURE_OPENAI_API_VERSION", default="2024-06-01"),
    )

    if main_provider == "azure":
        main_profile.api_key = _get_value(values, "API_KEY", "AZURE_OPENAI_API_KEY")
        main_profile.azure_endpoint = _get_value(values, "AZURE_ENDPOINT", "AZURE_OPENAI_ENDPOINT")
    elif main_provider in OPENAI_COMPATIBLE_PROVIDERS:
        main_profile.api_key = _resolve_openai_like_api_key(values)
        main_profile.base_url = _resolve_openai_like_base_url(values)
    elif main_provider == "anthropic":
        main_profile.api_key = _get_value(values, "ANTHROPIC_API_KEY")

    if scope == "main":
        return main_profile

    inherit_main = env_truthy(_get_value(values, "OCR_LLM_INHERIT_MAIN", default="1"))
    if inherit_main:
        return LLMProfile(
            scope="ocr",
            provider=main_profile.provider,
            model=main_profile.model,
            api_key=main_profile.api_key,
            base_url=main_profile.base_url,
            azure_endpoint=main_profile.azure_endpoint,
            api_version=main_profile.api_version,
            inherit_main=True,
        )

    provider = _get_value(values, "OCR_LLM_PROVIDER", default=main_profile.provider).strip().lower() or main_profile.provider
    model = _get_value(values, "OCR_LLM_MODEL", default=main_profile.model).strip() or main_profile.model

    profile = LLMProfile(
        scope="ocr",
        provider=provider,
        model=model,
        inherit_main=False,
        api_version=_get_value(
            values,
            "OCR_API_VERSION",
            "OCR_AZURE_OPENAI_API_VERSION",
            "API_VERSION",
            "AZURE_OPENAI_API_VERSION",
            default=main_profile.api_version or "2024-06-01",
        ),
    )

    if provider == "azure":
        profile.api_key = _get_value(values, "OCR_API_KEY", "OCR_AZURE_OPENAI_API_KEY", "API_KEY", "AZURE_OPENAI_API_KEY")
        profile.azure_endpoint = _get_value(values, "OCR_AZURE_ENDPOINT", "OCR_AZURE_OPENAI_ENDPOINT", "AZURE_ENDPOINT", "AZURE_OPENAI_ENDPOINT")
    elif provider in OPENAI_COMPATIBLE_PROVIDERS:
        profile.api_key = _resolve_openai_like_api_key(values, prefix="OCR_")
        profile.base_url = _resolve_openai_like_base_url(values, prefix="OCR_")
    elif provider == "anthropic":
        profile.api_key = _get_value(values, "OCR_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")

    return profile


def list_available_models(profile: LLMProfile, *, limit: int = 500) -> ModelCatalogResult:
    provider = profile.provider

    if provider in MANUAL_MODEL_LIST_PROVIDERS:
        return ModelCatalogResult(
            ok=False,
            provider=provider,
            models=[],
            manual_only=True,
            status="Azure model listing stays manual in this pass. Enter the deployment/model name directly.",
        )

    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        kwargs = {"api_key": profile.api_key or "EMPTY"}
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        client = OpenAI(**kwargs)
        model_list = client.models.list()
        models = []
        for item in getattr(model_list, "data", []):
            model_id = getattr(item, "id", None)
            if not model_id:
                continue
            models.append(ModelOption(id=model_id, label=model_id, provider=provider, raw=item))
        models.sort(key=lambda item: item.id)
        return ModelCatalogResult(
            ok=True,
            provider=provider,
            models=models[:limit],
            status=f"Fetched {len(models[:limit])} model(s) from {provider}.",
        )

    if provider == "anthropic":
        if Anthropic is None:
            raise ImportError("anthropic package is not installed. Please install anthropic.")
        client = Anthropic(api_key=profile.api_key)
        models = []
        for index, item in enumerate(client.models.list()):
            if index >= limit:
                break
            model_id = getattr(item, "id", None)
            if not model_id:
                continue
            display_name = getattr(item, "display_name", None) or model_id
            label = display_name if display_name == model_id else f"{display_name} ({model_id})"
            models.append(ModelOption(id=model_id, label=label, provider=provider, raw=item))
        models.sort(key=lambda item: item.label.lower())
        return ModelCatalogResult(
            ok=True,
            provider=provider,
            models=models,
            status=f"Fetched {len(models)} model(s) from anthropic.",
        )

    raise ValueError(f"Unsupported provider for model listing: {provider}")
