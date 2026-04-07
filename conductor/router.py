from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Tier(str, Enum):
    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"
    LOCAL = "local"


@dataclass
class RoutingDecision:
    tier: Tier
    score: float
    reasoning: str
    confidence: float
    tokens_in: int = 0


@dataclass
class TaskProfile:
    task_pattern: str
    call_count: int = 0
    tier_counts: dict = field(default_factory=dict)
    avg_score: float = 0.0

    def dominant_tier(self) -> Optional[Tier]:
        if not self.tier_counts or self.call_count < 5:
            return None
        top = max(self.tier_counts, key=self.tier_counts.get)
        if self.tier_counts[top] / self.call_count >= 0.75:
            return Tier(top)
        return None


_ROUTING_PROMPT = """\
You are a task router for a tiered AI system. Score this task and assign it to the correct tier.

TASK:
{task}

CONTEXT SUMMARY (if any):
{context}

Tiers:
- LOCAL: Pure execution, formatting, regex, calculations. No reasoning required.
- HAIKU: Classification, simple Q&A, diagnostics, short summaries.
- SONNET: Research, drafting, analysis, multi-step reasoning, code generation.
- OPUS: High-judgment decisions with ambiguous evidence. Legal/financial/medical nuance.
  Strategic decisions where errors are costly. Only use when nuance genuinely matters.

Return JSON only:
{{
  "tier": "local|haiku|sonnet|opus",
  "score": 0.0-1.0,
  "reasoning": "one sentence",
  "confidence": 0.0-1.0
}}"""


class TaskRouter:
    """
    Routes tasks to the appropriate model tier via a Haiku scoring call.

    After a task pattern accumulates enough history, the router skips the
    scoring call and uses the cached decision instead.
    """

    def __init__(self, client, session_history: bool = True):
        self._client = client
        self._profiles: dict[str, TaskProfile] = {}
        self._session_history = session_history
        self._routing_calls = 0
        self._routing_tokens = 0
        self._cache_hits = 0

    def route(
        self,
        task: str,
        context_summary: str = "",
        force_tier: Optional[Tier] = None,
    ) -> RoutingDecision:
        if force_tier:
            return RoutingDecision(
                tier=force_tier,
                score=0.5,
                reasoning="Forced by caller",
                confidence=1.0,
            )

        if self._session_history:
            cached = self._check_cache(task)
            if cached:
                self._cache_hits += 1
                return cached

        prompt = _ROUTING_PROMPT.format(
            task=task[:500],
            context=context_summary[:200] if context_summary else "None",
        )

        try:
            response = self._client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            self._routing_calls += 1
            self._routing_tokens += (
                response.usage.input_tokens + response.usage.output_tokens
            )

            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            data = json.loads(raw.strip())
            decision = RoutingDecision(
                tier=Tier(data["tier"]),
                score=float(data.get("score", 0.5)),
                reasoning=data.get("reasoning", ""),
                confidence=float(data.get("confidence", 0.8)),
                tokens_in=response.usage.input_tokens,
            )
            self._update_profile(task, decision)
            return decision

        except Exception as exc:
            return RoutingDecision(
                tier=Tier.SONNET,
                score=0.5,
                reasoning=f"Routing error (defaulting to sonnet): {exc}",
                confidence=0.5,
            )

    def _check_cache(self, task: str) -> Optional[RoutingDecision]:
        key = _task_key(task)
        profile = self._profiles.get(key)
        if not profile:
            return None
        dominant = profile.dominant_tier()
        if not dominant:
            return None
        n = profile.tier_counts.get(dominant.value, 0)
        return RoutingDecision(
            tier=dominant,
            score=profile.avg_score,
            reasoning=f"Cached ({profile.call_count} calls, {n} → {dominant.value})",
            confidence=0.95,
        )

    def _update_profile(self, task: str, decision: RoutingDecision) -> None:
        key = _task_key(task)
        profile = self._profiles.setdefault(key, TaskProfile(task_pattern=key))
        profile.call_count += 1
        profile.tier_counts[decision.tier.value] = (
            profile.tier_counts.get(decision.tier.value, 0) + 1
        )
        profile.avg_score = (
            profile.avg_score * (profile.call_count - 1) + decision.score
        ) / profile.call_count

    def stats(self) -> dict:
        total = self._routing_calls + self._cache_hits
        return {
            "routing_calls": self._routing_calls,
            "routing_tokens": self._routing_tokens,
            "cache_hits": self._cache_hits,
            "profiles": len(self._profiles),
            "cache_hit_rate": self._cache_hits / max(1, total),
        }


def _task_key(task: str) -> str:
    return " ".join(task.lower().split()[:8])
