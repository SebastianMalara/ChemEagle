from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI, BadRequestError, OpenAI

from llm_profiles import OPENAI_COMPATIBLE_PROVIDERS, LLMProfile, resolve_llm_profile

try:
    from anthropic import Anthropic
except ImportError:  # optional dependency
    Anthropic = None


@dataclass
class ToolCallFunction:
    name: str
    arguments: str = "{}"

    def model_dump(self, **_: Any) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass
class ChatToolCall:
    id: str
    function: ToolCallFunction
    type: str = "function"

    def model_dump(self, **_: Any) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.model_dump(),
        }


@dataclass
class ChatMessage:
    role: str
    content: Any = None
    tool_calls: List[ChatToolCall] = field(default_factory=list)

    def model_dump(self, **_: Any) -> Dict[str, Any]:
        payload = {"role": self.role}
        if self.content is not None:
            payload["content"] = self.content
        if self.tool_calls:
            payload["tool_calls"] = [tool_call.model_dump() for tool_call in self.tool_calls]
        return payload


@dataclass
class ChatChoice:
    message: ChatMessage


@dataclass
class ChatCompletionResponse:
    choices: List[ChatChoice]
    raw: Any = None


class _ChatCompletionsAdapter:
    def __init__(self, wrapper: "LLMWrapper"):
        self.wrapper = wrapper

    def create(self, **kwargs: Any) -> ChatCompletionResponse:
        return self.wrapper.chat_completion(**kwargs)


class _ChatAdapter:
    def __init__(self, wrapper: "LLMWrapper"):
        self.completions = _ChatCompletionsAdapter(wrapper)


class CompatibleChatClient:
    def __init__(self, wrapper: "LLMWrapper"):
        self.chat = _ChatAdapter(wrapper)


class LLMWrapper:
    """Provider-aware chat wrapper that exposes an OpenAI-style chat interface."""

    def __init__(self, profile: LLMProfile, client: Any):
        self.profile = profile
        self.provider = profile.provider
        self.model = profile.model
        self.client = client
        self.chat = _ChatAdapter(self)

    @classmethod
    def from_env(
        cls,
        default_model: str = "gpt-5-mini",
        *,
        scope: str = "main",
        values: Optional[Dict[str, Any]] = None,
    ) -> "LLMWrapper":
        profile = resolve_llm_profile(scope=scope, values=values, default_model=default_model)
        return cls.from_profile(profile)

    @classmethod
    def from_profile(cls, profile: LLMProfile) -> "LLMWrapper":
        if profile.provider == "azure":
            if not profile.api_key or not profile.azure_endpoint:
                raise ValueError("Azure provider requires API_KEY and AZURE_ENDPOINT")
            client = AzureOpenAI(
                api_key=profile.api_key,
                api_version=profile.api_version or "2024-06-01",
                azure_endpoint=profile.azure_endpoint,
            )
            return cls(profile=profile, client=client)

        if profile.provider in OPENAI_COMPATIBLE_PROVIDERS:
            if profile.provider == "openai" and not profile.api_key and not profile.base_url:
                raise ValueError("OpenAI provider requires OPENAI_API_KEY (or API_KEY)")

            kwargs = {"api_key": profile.api_key or "EMPTY"}
            if profile.base_url:
                kwargs["base_url"] = profile.base_url
            client = OpenAI(**kwargs)
            return cls(profile=profile, client=client)

        if profile.provider == "anthropic":
            if Anthropic is None:
                raise ImportError("anthropic package is not installed. Please install anthropic")
            if not profile.api_key:
                raise ValueError("Anthropic provider requires ANTHROPIC_API_KEY")
            client = Anthropic(api_key=profile.api_key)
            return cls(profile=profile, client=client)

        raise ValueError(f"Unsupported LLM provider: {profile.provider}")

    def as_chat_completion_client(self) -> CompatibleChatClient:
        return CompatibleChatClient(self)

    def chat_completion_text(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        response = self.chat_completion(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    def chat_completion(
        self,
        *,
        model: Optional[str] = None,
        messages: List[Dict[str, Any]],
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatCompletionResponse:
        model_name = model or self.model

        if self.provider in OPENAI_COMPATIBLE_PROVIDERS or self.provider == "azure":
            kwargs: Dict[str, Any] = {
                "model": model_name,
                "messages": messages,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if response_format is not None:
                kwargs["response_format"] = response_format
            if tools is not None:
                kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
            response = self._create_openai_chat_with_fallback(kwargs)
            return self._normalize_openai_response(response)

        anthropic_messages, system_prompt = self._to_anthropic_messages(messages, response_format=response_format)
        kwargs = {
            "model": model_name,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or int(os.getenv("LLM_MAX_TOKENS", "2048")),
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = self._to_anthropic_tools(tools)
        if tool_choice:
            kwargs["tool_choice"] = self._to_anthropic_tool_choice(tool_choice)
        response = self.client.messages.create(**kwargs)
        return self._normalize_anthropic_response(response)

    def _create_openai_chat_with_fallback(self, kwargs: Dict[str, Any]):
        try:
            return self.client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            retry_kwargs = dict(kwargs)
            changed = False
            if "temperature" in retry_kwargs and self._is_temperature_error(exc):
                retry_kwargs.pop("temperature", None)
                changed = True
            if "response_format" in retry_kwargs and self._is_response_format_error(exc):
                retry_kwargs.pop("response_format", None)
                changed = True
            if changed:
                return self.client.chat.completions.create(**retry_kwargs)
            raise
        except Exception as exc:
            retry_kwargs = dict(kwargs)
            changed = False
            error_text = str(exc).lower()
            if "response_format" in retry_kwargs and "response_format" in error_text:
                retry_kwargs.pop("response_format", None)
                changed = True
            if changed:
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
    def _is_response_format_error(exc: BadRequestError) -> bool:
        err = getattr(exc, "body", {}) or {}
        detail = err.get("error", {}) if isinstance(err, dict) else {}
        if detail.get("param") == "response_format":
            return True
        msg = str(exc).lower()
        return "response_format" in msg

    def _normalize_openai_response(self, response: Any) -> ChatCompletionResponse:
        message = response.choices[0].message
        normalized_tool_calls: List[ChatToolCall] = []
        for tool_call in getattr(message, "tool_calls", None) or []:
            function_payload = getattr(tool_call, "function", None)
            normalized_tool_calls.append(
                ChatToolCall(
                    id=getattr(tool_call, "id", ""),
                    type=getattr(tool_call, "type", "function"),
                    function=ToolCallFunction(
                        name=getattr(function_payload, "name", ""),
                        arguments=getattr(function_payload, "arguments", "{}") or "{}",
                    ),
                )
            )
        normalized_message = ChatMessage(
            role=getattr(message, "role", "assistant"),
            content=getattr(message, "content", None),
            tool_calls=normalized_tool_calls,
        )
        return ChatCompletionResponse(choices=[ChatChoice(message=normalized_message)], raw=response)

    def _normalize_anthropic_response(self, response: Any) -> ChatCompletionResponse:
        text_parts: List[str] = []
        tool_calls: List[ChatToolCall] = []

        for block in getattr(response, "content", None) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))
            elif block_type == "tool_use":
                tool_calls.append(
                    ChatToolCall(
                        id=getattr(block, "id", ""),
                        function=ToolCallFunction(
                            name=getattr(block, "name", ""),
                            arguments=json.dumps(getattr(block, "input", {}) or {}, ensure_ascii=False),
                        ),
                    )
                )

        normalized_message = ChatMessage(
            role="assistant",
            content="\n".join(part for part in text_parts if part).strip(),
            tool_calls=tool_calls,
        )
        return ChatCompletionResponse(choices=[ChatChoice(message=normalized_message)], raw=response)

    def _to_anthropic_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[Dict[str, Any]], str]:
        converted: List[Dict[str, Any]] = []
        system_parts: List[str] = []
        pending_tool_results: List[Dict[str, Any]] = []

        def flush_pending_tool_results() -> None:
            if pending_tool_results:
                converted.append({"role": "user", "content": list(pending_tool_results)})
                pending_tool_results.clear()

        for raw_message in messages:
            message = self._message_to_dict(raw_message)
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "tool":
                pending_tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id", ""),
                        "content": str(content),
                    }
                )
                continue

            flush_pending_tool_results()

            if role == "system":
                system_parts.append(self._content_to_text(content))
                continue

            if role not in {"user", "assistant"}:
                role = "user"

            blocks = self._convert_content_blocks(content)
            if role == "assistant" and message.get("tool_calls"):
                blocks.extend(self._assistant_tool_calls_to_anthropic_blocks(message["tool_calls"]))

            converted.append(
                {
                    "role": role,
                    "content": blocks or [{"type": "text", "text": ""}],
                }
            )

        flush_pending_tool_results()

        if response_format and response_format.get("type") == "json_object":
            system_parts.append("Return valid JSON only. Do not wrap it in markdown and do not add any extra explanation.")

        system_prompt = "\n\n".join(part for part in system_parts if part)
        return converted, system_prompt

    @staticmethod
    def _assistant_tool_calls_to_anthropic_blocks(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        for raw_tool_call in tool_calls:
            tool_call = LLMWrapper._tool_call_to_dict(raw_tool_call)
            function_payload = tool_call.get("function", {})
            arguments = function_payload.get("arguments", "{}") or "{}"
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = {}

            blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id", ""),
                    "name": function_payload.get("name", ""),
                    "input": parsed_arguments,
                }
            )
        return blocks

    @staticmethod
    def _to_anthropic_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for tool in tools:
            function_payload = tool.get("function", tool)
            converted.append(
                {
                    "name": function_payload.get("name", ""),
                    "description": function_payload.get("description", ""),
                    "input_schema": function_payload.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return converted

    @staticmethod
    def _to_anthropic_tool_choice(tool_choice: Any) -> Dict[str, Any]:
        if isinstance(tool_choice, dict):
            function_payload = tool_choice.get("function", {})
            tool_name = function_payload.get("name")
            if tool_choice.get("type") == "function" and tool_name:
                return {"type": "tool", "name": tool_name}
            return tool_choice

        if tool_choice == "required":
            return {"type": "any"}
        if tool_choice == "none":
            return {"type": "auto"}
        return {"type": "auto"}

    @staticmethod
    def _message_to_dict(message: Any) -> Dict[str, Any]:
        if isinstance(message, dict):
            return message
        if hasattr(message, "model_dump"):
            dumped = message.model_dump()
            if isinstance(dumped, dict):
                return dumped
        payload = {
            "role": getattr(message, "role", "user"),
            "content": getattr(message, "content", ""),
        }
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            payload["tool_calls"] = [LLMWrapper._tool_call_to_dict(tool_call) for tool_call in tool_calls]
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            payload["tool_call_id"] = tool_call_id
        return payload

    @staticmethod
    def _tool_call_to_dict(tool_call: Any) -> Dict[str, Any]:
        if isinstance(tool_call, dict):
            return tool_call
        if hasattr(tool_call, "model_dump"):
            dumped = tool_call.model_dump()
            if isinstance(dumped, dict):
                return dumped
        function_payload = getattr(tool_call, "function", None)
        return {
            "id": getattr(tool_call, "id", ""),
            "type": getattr(tool_call, "type", "function"),
            "function": {
                "name": getattr(function_payload, "name", ""),
                "arguments": getattr(function_payload, "arguments", "{}") or "{}",
            },
        }

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ).strip()
        return str(content)

    @staticmethod
    def _convert_content_blocks(content: Any) -> List[Dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]

        blocks: List[Dict[str, Any]] = []
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    blocks.append({"type": "text", "text": str(item)})
                    continue

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
                    elif url:
                        blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": url,
                                },
                            }
                        )
                else:
                    blocks.append({"type": "text", "text": str(item)})
        else:
            blocks.append({"type": "text", "text": str(content)})

        return blocks or [{"type": "text", "text": ""}]
