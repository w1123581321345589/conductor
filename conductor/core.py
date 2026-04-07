from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from conductor.brief import Brief, BriefBuilder
from conductor.local import LocalExecutor
from conductor.router import RoutingDecision, TaskRouter, Tier

MODELS = {
    Tier.HAIKU: "claude-haiku-4-5",
    Tier.SONNET: "claude-sonnet-4-5",
    Tier.OPUS: "claude-opus-4-5",
}

# Blended cost per 1M tokens in USD (input/output averaged)
COST_PER_1M = {
    Tier.HAIKU: 0.80,
    Tier.SONNET: 9.00,
    Tier.OPUS: 75.00,
    Tier.LOCAL: 0.00,
}


@dataclass
class ConductorResult:
    content: str
    tier: Tier
    routing: RoutingDecision
    brief: Optional[Brief]
    tokens_in: int
    tokens_out: int
    cost_usd: float
    from_local: bool = False


@dataclass
class SessionStats:
    total_calls: int = 0
    tier_calls: dict = field(default_factory=dict)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    tokens_saved_brief: int = 0
    cache_hits: int = 0

    @property
    def cost_vs_all_opus(self) -> float:
        return self.total_tokens * COST_PER_1M[Tier.OPUS] / 1_000_000

    @property
    def savings_pct(self) -> float:
        baseline = self.cost_vs_all_opus
        if baseline == 0:
            return 0.0
        return (1 - self.total_cost_usd / baseline) * 100


class Conductor:
    """
    Tiered AI orchestration over the Anthropic API.

    Each task is scored and routed to the cheapest model that can handle it.
    Large context is compressed by Haiku before going to Sonnet or Opus.
    Local models (via Ollama or LM Studio) handle execution tasks at $0/call.

    Basic usage::

        import anthropic
        from conductor import Conductor

        c = Conductor(client=anthropic.Anthropic())
        result = c.run("Summarize the key risks in this contract")
        print(result.content)
        print(f"tier={result.tier.value}  cost=${result.cost_usd:.4f}")

    With context compression::

        result = c.run("Find the three highest-risk items", context=long_doc)
        print(f"saved {result.brief.savings:,} tokens")

    With a local model::

        c = Conductor(client=client, local_model="qwen3:32b")
    """

    def __init__(
        self,
        client,
        local_model: Optional[str] = None,
        compress_context: bool = True,
        use_cache: bool = True,
        default_max_tokens: int = 2000,
    ):
        self._client = client
        self._default_max_tokens = default_max_tokens
        self._router = TaskRouter(client, session_history=use_cache)
        self._briefer = BriefBuilder(client) if compress_context else None
        self._local = (
            LocalExecutor(model=local_model, fallback_client=client)
            if local_model
            else None
        )
        self._stats = SessionStats()

    def run(
        self,
        task: str,
        context: str = "",
        force_tier: Optional[Tier] = None,
        max_tokens: Optional[int] = None,
        system: str = "",
    ) -> ConductorResult:
        """
        Route and execute a task.

        Args:
            task: The task to execute.
            context: Background context. Compressed automatically when large.
            force_tier: Skip routing and use this tier directly.
            max_tokens: Override the default output token limit.
            system: Optional system prompt for the model call.

        Returns:
            ConductorResult with content, tier, cost, and compression metadata.
        """
        max_tokens = max_tokens or self._default_max_tokens

        routing = self._router.route(
            task,
            context_summary=context[:500] if context else "",
            force_tier=force_tier,
        )

        brief = None
        effective_context = context
        if context and self._briefer and routing.tier in (Tier.SONNET, Tier.OPUS):
            brief = self._briefer.build(task, context)
            effective_context = brief.content

        if routing.tier == Tier.LOCAL and self._local:
            local_result = self._local.run(
                task=_build_prompt(task, effective_context),
                max_tokens=max_tokens,
            )
            result = ConductorResult(
                content=local_result.content,
                tier=Tier.LOCAL,
                routing=routing,
                brief=brief,
                tokens_in=0,
                tokens_out=local_result.tokens,
                cost_usd=0.0,
                from_local=local_result.from_local,
            )
        else:
            tier = routing.tier if routing.tier != Tier.LOCAL else Tier.SONNET
            model = MODELS[tier]
            messages = [{"role": "user", "content": _build_prompt(task, effective_context)}]
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            if system:
                kwargs["system"] = system

            response = self._client.messages.create(**kwargs)
            content = response.content[0].text
            tokens_in = response.usage.input_tokens
            tokens_out = response.usage.output_tokens
            cost = (tokens_in + tokens_out) * COST_PER_1M[tier] / 1_000_000

            result = ConductorResult(
                content=content,
                tier=tier,
                routing=routing,
                brief=brief,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
            )

        self._record(result, brief)
        return result

    def route_only(self, task: str, context: str = "") -> RoutingDecision:
        """Score a task without executing it."""
        return self._router.route(task, context_summary=context[:200])

    def stats(self) -> SessionStats:
        return self._stats

    def print_stats(self) -> None:
        s = self._stats
        r = self._router.stats()
        print("\nConductor Session Stats")
        print(f"  Total calls:       {s.total_calls}")
        print(f"  Tier distribution: {s.tier_calls}")
        print(f"  Total tokens:      {s.total_tokens:,}")
        print(f"  Total cost:        ${s.total_cost_usd:.4f}")
        print(f"  vs all-Opus:       ${s.cost_vs_all_opus:.4f}")
        print(f"  Savings:           {s.savings_pct:.0f}%")
        print(f"  Context compression saved: {s.tokens_saved_brief:,} tokens")
        print(
            f"  Router cache hits: {r['cache_hits']} "
            f"({r['cache_hit_rate'] * 100:.0f}%)"
        )

    def _record(self, result: ConductorResult, brief: Optional[Brief]) -> None:
        self._stats.total_calls += 1
        key = result.tier.value
        self._stats.tier_calls[key] = self._stats.tier_calls.get(key, 0) + 1
        self._stats.total_tokens += result.tokens_in + result.tokens_out
        self._stats.total_cost_usd += result.cost_usd
        if brief:
            self._stats.tokens_saved_brief += brief.savings


def _build_prompt(task: str, context: str) -> str:
    if not context:
        return task
    return f"Context:\n{context}\n\nTask:\n{task}"
