import os
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI, BadRequestError, OpenAI

try:
    from anthropic import Anthropic
except ImportError:  # optional dependency
    Anthropic = None


class LLMWrapper:
    """Unified chat wrapper for Azure/OpenAI-compatible/Anthropic providers."""

    def __init__(self, provider: str, model: str, client: Any):
        self.provider = provider
        self.model = model
        self.client = client

    @classmethod
    def from_env(cls, default_model: str = "gpt-5-mini") -> "LLMWrapper":
        provider = os.getenv("LLM_PROVIDER", "azure").strip().lower()
        model = os.getenv("LLM_MODEL", default_model)

        if provider == "azure":
            api_key = os.getenv("API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
            api_version = os.getenv("API_VERSION", "2024-06-01")
            if not api_key or not endpoint:
                raise ValueError("Azure provider requires API_KEY and AZURE_ENDPOINT")
            client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)
            return cls(provider=provider, model=model, client=client)

        if provider in {"openai", "openai_compatible", "lmstudio", "local_openai"}:
            # Supports OpenAI cloud and any OpenAI-compatible endpoint (LM Studio, vLLM, Ollama OpenAI API, etc.)
            base_url = (
                os.getenv("LLM_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or os.getenv("VLLM_BASE_URL")
                or os.getenv("LMSTUDIO_BASE_URL")
                or os.getenv("OLLAMA_BASE_URL")
            )
            api_key = (
                os.getenv("LLM_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("VLLM_API_KEY")
                or os.getenv("API_KEY")
                or os.getenv("LMSTUDIO_API_KEY")
                or os.getenv("OLLAMA_API_KEY")
            )

            # OpenAI cloud needs API key; local compatible servers often accept any non-empty key.
            if provider == "openai" and not api_key and not base_url:
                raise ValueError("OpenAI provider requires OPENAI_API_KEY (or API_KEY)")

            kwargs = {"api_key": api_key or "lm-studio"}
            if base_url:
                kwargs["base_url"] = base_url
            client = OpenAI(**kwargs)
            return cls(provider=provider, model=model, client=client)

        if provider == "anthropic":
            if Anthropic is None:
                raise ImportError("anthropic package is not installed. Please install anthropic")
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic provider requires ANTHROPIC_API_KEY")
            client = Anthropic(api_key=api_key)
            return cls(provider=provider, model=model, client=client)

        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

    def chat_completion_text(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
    ) -> str:
        model = model or self.model

        if self.provider in {"azure", "openai", "openai_compatible", "lmstudio", "local_openai"}:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if response_format is not None:
                kwargs["response_format"] = response_format
            response = self._create_chat_with_temperature_fallback(kwargs)
            return (response.choices[0].message.content or "").strip()

        anthropic_messages, system_prompt = self._to_anthropic_messages(messages)
        kwargs = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048")),
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)

        output_parts: List[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                output_parts.append(block.text)
        return "\n".join(output_parts).strip()

    def _create_chat_with_temperature_fallback(self, kwargs: Dict[str, Any]):
        """Retry once without `temperature` for models that only accept default temperature."""
        try:
            return self.client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            if self._is_temperature_error(exc) and "temperature" in kwargs:
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("temperature", None)
                return self.client.chat.completions.create(**retry_kwargs)
            raise

    @staticmethod
    def _is_temperature_error(exc: BadRequestError) -> bool:
        err = getattr(exc, "body", {}) or {}
        detail = err.get("error", {}) if isinstance(err, dict) else {}
        if detail.get("param") == "temperature":
            return True

        msg = str(exc).lower()
        return "temperature" in msg and "unsupported" in msg

    @staticmethod
    def _to_anthropic_messages(messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], str]:
        converted: List[Dict[str, Any]] = []
        system_parts: List[str] = []

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
                else:
                    system_parts.append(str(content))
                continue

            if role == "tool":
                role = "user"
                tool_name = m.get("name", "tool")
                content = f"Tool output ({tool_name}):\n{content}"

            if role not in {"user", "assistant"}:
                role = "user"

            converted.append({"role": role, "content": LLMWrapper._convert_content_blocks(content)})

        return converted, "\n\n".join(system_parts)

    @staticmethod
    def _convert_content_blocks(content: Any) -> List[Dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]

        blocks: List[Dict[str, Any]] = []
        if isinstance(content, list):
            for item in content:
                item_type = item.get("type")
                if item_type == "text":
                    blocks.append({"type": "text", "text": item.get("text", "")})
                elif item_type == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image") and "," in url:
                        header, data = url.split(",", 1)
                        media_type = header.split(";")[0].replace("data:", "")
                        blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data,
                                },
                            }
                        )
                else:
                    blocks.append({"type": "text", "text": str(item)})
        else:
            blocks.append({"type": "text", "text": str(content)})

        return blocks or [{"type": "text", "text": ""}]
