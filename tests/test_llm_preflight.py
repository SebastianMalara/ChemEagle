from __future__ import annotations

import unittest
from unittest import mock

from llm_preflight import (
    ProviderProbeResult,
    RunFailureController,
    _create_openai_probe_completion,
    classify_provider_exception,
    collect_runtime_provider_preflight,
)


class LlmPreflightTests(unittest.TestCase):
    def test_classify_connection_error_as_systemic_dns_failure(self) -> None:
        failure = classify_provider_exception(RuntimeError("Connection error."))
        self.assertEqual(failure.kind, "dns_or_connection_error")
        self.assertTrue(failure.systemic)
        self.assertFalse(failure.retryable)

    def test_run_failure_controller_aborts_immediately_on_first_source_connection_failure(self) -> None:
        controller = RunFailureController()
        failure = classify_provider_exception(RuntimeError("Connection error."))
        aborted, reason = controller.record(failure, source_index=0, source_name="paper.pdf")
        self.assertTrue(aborted)
        self.assertIn("paper.pdf", reason)
        self.assertEqual(controller.state.systemic_failure_kind, "dns_or_connection_error")
        self.assertEqual(controller.state.systemic_failure_count, 1)

    def test_collect_runtime_provider_preflight_reports_blocking_probe_failures(self) -> None:
        def fake_probe(profile, purpose):
            ok = purpose == "main_text"
            return ProviderProbeResult(
                ok=ok,
                purpose=purpose,
                provider=profile.provider,
                model=profile.model,
                failure_kind="" if ok else "unsupported_model_or_capability",
                message="ok" if ok else "vision unsupported",
                http_status=None,
                supports_images=None if purpose == "main_text" else False,
                diagnostic_payload={"purpose": purpose},
            )

        config = {
            "CHEMEAGLE_RUN_MODE": "cloud",
            "LLM_PROVIDER": "openai_compatible",
            "LLM_MODEL": "gpt-5-mini",
            "OPENAI_BASE_URL": "http://localhost:8000/v1",
            "OPENAI_API_KEY": "EMPTY",
            "OCR_BACKEND": "llm_vision",
            "OCR_LLM_INHERIT_MAIN": "0",
            "OCR_LLM_PROVIDER": "openai_compatible",
            "OCR_LLM_MODEL": "vision-model",
            "OCR_OPENAI_BASE_URL": "http://localhost:8000/v1",
            "OCR_OPENAI_API_KEY": "EMPTY",
        }
        with mock.patch("llm_preflight.probe_llm_profile", side_effect=fake_probe):
            diagnostics = collect_runtime_provider_preflight(profile_configs=[config], mode="cloud")

        self.assertEqual(diagnostics["status"], "failed_blocking")
        self.assertEqual(len(diagnostics["results"]), 2)
        self.assertEqual(len(diagnostics["blocking_errors"]), 1)
        self.assertIn("vision unsupported", diagnostics["blocking_errors"][0])

    def test_unsupported_parameter_is_not_misclassified_as_model_capability(self) -> None:
        failure = classify_provider_exception(
            RuntimeError(
                "Error code: 400 - {'error': {'message': \"Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.\", 'type': 'invalid_request_error', 'param': 'max_tokens', 'code': 'unsupported_parameter'}}"
            )
        )
        self.assertEqual(failure.kind, "bad_request_non_retryable")
        self.assertFalse(failure.systemic)

    def test_openai_probe_falls_back_from_max_completion_tokens_to_max_tokens(self) -> None:
        class FakeChatCompletions:
            def __init__(self) -> None:
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                if "max_completion_tokens" in kwargs:
                    raise RuntimeError("Unsupported parameter: max_completion_tokens")
                return {"ok": True}

        fake_client = mock.Mock()
        fake_client.chat.completions = FakeChatCompletions()
        result = _create_openai_probe_completion(
            client=fake_client,
            model="test-model",
            messages=[{"role": "user", "content": "Reply with OK."}],
        )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(len(fake_client.chat.completions.calls), 2)
        self.assertIn("max_completion_tokens", fake_client.chat.completions.calls[0])
        self.assertIn("max_tokens", fake_client.chat.completions.calls[1])


if __name__ == "__main__":
    unittest.main()
