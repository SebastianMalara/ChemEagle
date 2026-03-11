from __future__ import annotations

import inspect
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from review_pricing import estimate_cost_usd


@dataclass
class LLMCallMetric:
    phase: str
    provider: str
    model: str
    usage_prompt_tokens: Optional[int]
    usage_completion_tokens: Optional[int]
    usage_total_tokens: Optional[int]
    estimated_cost_usd: Optional[float]
    latency_ms: int
    success: bool
    raw_usage_json: str

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunMetricsCollector:
    calls: List[LLMCallMetric] = field(default_factory=list)

    def record(
        self,
        *,
        phase: str,
        provider: str,
        model: str,
        usage: Dict[str, Optional[int]],
        latency_ms: int,
        success: bool,
        raw_usage: Dict[str, Any],
    ) -> None:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        self.calls.append(
            LLMCallMetric(
                phase=phase,
                provider=provider,
                model=model,
                usage_prompt_tokens=prompt_tokens,
                usage_completion_tokens=completion_tokens,
                usage_total_tokens=total_tokens,
                estimated_cost_usd=estimate_cost_usd(provider, model, prompt_tokens, completion_tokens),
                latency_ms=latency_ms,
                success=success,
                raw_usage_json=json.dumps(raw_usage or {}, ensure_ascii=False),
            )
        )

    def summary(self) -> Dict[str, Any]:
        prompt_tokens = sum(call.usage_prompt_tokens or 0 for call in self.calls)
        completion_tokens = sum(call.usage_completion_tokens or 0 for call in self.calls)
        total_tokens = sum(call.usage_total_tokens or 0 for call in self.calls)
        known_costs = [call.estimated_cost_usd for call in self.calls if call.estimated_cost_usd is not None]
        usage_count = sum(1 for call in self.calls if call.usage_total_tokens is not None)
        if not self.calls:
            usage_completeness = "none"
        elif usage_count == len(self.calls):
            usage_completeness = "complete"
        elif usage_count == 0:
            usage_completeness = "none"
        else:
            usage_completeness = "partial"
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(sum(known_costs), 8) if known_costs else None,
            "usage_completeness": usage_completeness,
            "call_count": len(self.calls),
        }


_CURRENT_COLLECTOR: ContextVar[Optional[RunMetricsCollector]] = ContextVar("review_current_collector", default=None)
_CURRENT_PHASE: ContextVar[str] = ContextVar("review_current_phase", default="")


@contextmanager
def bind_metrics_collector(collector: RunMetricsCollector) -> Iterator[RunMetricsCollector]:
    token = _CURRENT_COLLECTOR.set(collector)
    try:
        yield collector
    finally:
        _CURRENT_COLLECTOR.reset(token)


@contextmanager
def llm_phase(phase: str) -> Iterator[None]:
    token = _CURRENT_PHASE.set((phase or "").strip())
    try:
        yield
    finally:
        _CURRENT_PHASE.reset(token)


def current_collector() -> Optional[RunMetricsCollector]:
    return _CURRENT_COLLECTOR.get()


def current_phase(default: str = "unknown") -> str:
    configured = _CURRENT_PHASE.get().strip()
    if configured:
        return configured
    for frame in inspect.stack()[2:]:
        path = (frame.filename or "").replace("\\", "/")
        if path.endswith("/llm_wrapper.py"):
            continue
        module = frame.frame.f_globals.get("__name__", "")
        func = frame.function
        if module:
            return f"{module}.{func}"
        return func
    return default


def extract_usage_payload(raw_response: Any) -> Dict[str, Any]:
    usage = getattr(raw_response, "usage", None)
    if usage is None and isinstance(raw_response, dict):
        usage = raw_response.get("usage")
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except TypeError:
            pass
    if hasattr(usage, "to_dict"):
        return usage.to_dict()
    if isinstance(usage, dict):
        return usage
    payload: Dict[str, Any] = {}
    for attr in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"):
        if hasattr(usage, attr):
            payload[attr] = getattr(usage, attr)
    return payload


def normalize_usage(payload: Dict[str, Any]) -> Dict[str, Optional[int]]:
    prompt = payload.get("prompt_tokens")
    completion = payload.get("completion_tokens")
    total = payload.get("total_tokens")
    if prompt is None and payload.get("input_tokens") is not None:
        prompt = payload.get("input_tokens")
    if completion is None and payload.get("output_tokens") is not None:
        completion = payload.get("output_tokens")
    if total is None and prompt is not None and completion is not None:
        total = int(prompt) + int(completion)
    return {
        "prompt_tokens": int(prompt) if prompt is not None else None,
        "completion_tokens": int(completion) if completion is not None else None,
        "total_tokens": int(total) if total is not None else None,
    }


@contextmanager
def timed_call() -> Iterator[callable]:
    started = time.perf_counter()
    yield lambda: int((time.perf_counter() - started) * 1000)
