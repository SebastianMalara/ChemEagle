from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class PricingEntry:
    prompt_per_million: float
    completion_per_million: float


_PRICING_TABLE: Dict[str, Dict[str, PricingEntry]] = {
    "azure": {
        "gpt-5-mini": PricingEntry(prompt_per_million=0.25, completion_per_million=2.0),
        "gpt-4o-mini": PricingEntry(prompt_per_million=0.15, completion_per_million=0.6),
        "gpt-4.1-mini": PricingEntry(prompt_per_million=0.4, completion_per_million=1.6),
    },
    "openai": {
        "gpt-5-mini": PricingEntry(prompt_per_million=0.25, completion_per_million=2.0),
        "gpt-4o-mini": PricingEntry(prompt_per_million=0.15, completion_per_million=0.6),
        "gpt-4.1-mini": PricingEntry(prompt_per_million=0.4, completion_per_million=1.6),
    },
    "anthropic": {
        "claude-3-5-sonnet-latest": PricingEntry(prompt_per_million=3.0, completion_per_million=15.0),
        "claude-3-7-sonnet-latest": PricingEntry(prompt_per_million=3.0, completion_per_million=15.0),
    },
    "openai_compatible": {},
    "lmstudio": {},
    "local_openai": {},
    "vllm": {},
}


def get_pricing(provider: str, model: str) -> Optional[PricingEntry]:
    provider_key = (provider or "").strip().lower()
    model_key = (model or "").strip()
    provider_table = _PRICING_TABLE.get(provider_key)
    if not provider_table:
        return None
    return provider_table.get(model_key)


def estimate_cost_usd(
    provider: str,
    model: str,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
) -> Optional[float]:
    pricing = get_pricing(provider, model)
    if pricing is None:
        return None
    prompt = int(prompt_tokens or 0)
    completion = int(completion_tokens or 0)
    return round(
        (prompt / 1_000_000.0) * pricing.prompt_per_million
        + (completion / 1_000_000.0) * pricing.completion_per_million,
        8,
    )
