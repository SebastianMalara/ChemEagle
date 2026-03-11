from __future__ import annotations

import json
import unittest
from unittest import mock

from llm_profiles import LLMProfile
from llm_wrapper import LLMWrapper
from review_tracking import RunMetricsCollector, bind_metrics_collector, llm_phase
from runtime_guards import RuntimeStageError


class _FailingCompletions:
    def __init__(self, exc: Exception):
        self._exc = exc

    def create(self, **_: object):
        raise self._exc


class _FailingChat:
    def __init__(self, exc: Exception):
        self.completions = _FailingCompletions(exc)


class _FailingClient:
    def __init__(self, exc: Exception):
        self.chat = _FailingChat(exc)


class _EmptyChoicesCompletions:
    def create(self, **_: object):
        return type("Response", (), {"choices": []})()


class _EmptyChoicesClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _EmptyChoicesCompletions()})()


class LLMWrapperTests(unittest.TestCase):
    def test_from_profile_openai_uses_explicit_official_base_url_when_blank(self) -> None:
        profile = LLMProfile(
            scope="main",
            provider="openai",
            model="gpt-5-mini",
            api_key="test-key",
            base_url="",
        )

        with mock.patch("llm_wrapper.OpenAI") as mock_openai:
            mock_openai.return_value = object()
            LLMWrapper.from_profile(profile)

        _, kwargs = mock_openai.call_args
        self.assertEqual(kwargs["api_key"], "test-key")
        self.assertEqual(kwargs["base_url"], "https://api.openai.com/v1")

    def test_failure_attaches_diagnostics_and_records_failure_metric(self) -> None:
        try:
            raise ConnectionError("socket closed")
        except ConnectionError as cause:
            try:
                raise RuntimeError("Connection error.") from cause
            except RuntimeError as exc:
                wrapped_exc = exc

        profile = LLMProfile(
            scope="main",
            provider="openai",
            model="gpt-5-mini",
            api_key="test-key",
        )
        wrapper = LLMWrapper(profile=profile, client=_FailingClient(wrapped_exc))
        collector = RunMetricsCollector()

        with bind_metrics_collector(collector), llm_phase("planner"):
            with self.assertRaises(RuntimeError) as raised:
                wrapper.chat_completion(
                    model="gpt-5-mini",
                    messages=[{"role": "user", "content": "Hello"}],
                )

        diagnostics = getattr(raised.exception, "_provider_diagnostics", {})
        self.assertEqual(diagnostics["provider"], "openai")
        self.assertEqual(diagnostics["model"], "gpt-5-mini")
        self.assertEqual(diagnostics["base_url"], "https://api.openai.com/v1")
        self.assertEqual(diagnostics["llm_phase"], "planner")
        self.assertEqual(diagnostics["llm_stage"], "planner")
        self.assertEqual(diagnostics["exception_class"], "RuntimeError")
        self.assertEqual(diagnostics["cause_class"], "ConnectionError")
        self.assertIn("Connection error.", diagnostics["traceback"])

        self.assertEqual(len(collector.calls), 1)
        call = collector.calls[0]
        self.assertFalse(call.success)
        raw_usage = json.loads(call.raw_usage_json)
        self.assertEqual(raw_usage["failure_kind"], "dns_or_connection_error")
        self.assertEqual(raw_usage["diagnostics"]["llm_stage"], "planner")

    def test_empty_choices_raise_runtime_stage_error_with_retry_hint(self) -> None:
        profile = LLMProfile(
            scope="main",
            provider="openai",
            model="gpt-5-mini",
            api_key="test-key",
        )
        wrapper = LLMWrapper(profile=profile, client=_EmptyChoicesClient())

        with llm_phase("planner"), self.assertRaises(RuntimeStageError) as raised:
            wrapper.chat_completion_text(messages=[{"role": "user", "content": "Hello"}])

        self.assertIn("response choices", str(raised.exception))
        self.assertEqual(getattr(raised.exception, "_retry_trigger", ""), "auto_no_agents_retry")


if __name__ == "__main__":
    unittest.main()
