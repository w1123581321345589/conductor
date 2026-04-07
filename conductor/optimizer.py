from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class PromptVariant:
    template: str
    token_count: int
    quality_score: float
    test_count: int = 0
    wins: int = 0

    @property
    def efficiency(self) -> float:
        if self.token_count == 0:
            return 0.0
        return self.quality_score / self.token_count

    @property
    def win_rate(self) -> float:
        if self.test_count == 0:
            return 0.0
        return self.wins / self.test_count


@dataclass
class OptimizationResult:
    original_template: str
    best_template: str
    original_tokens: int
    best_tokens: int
    quality_maintained: bool
    token_reduction: float
    variants_tested: int


_VARIATION_PROMPT = """\
You are a prompt engineer optimizing for token efficiency.

ORIGINAL PROMPT TEMPLATE:
{template}

Rewrite this prompt to use fewer tokens while preserving its full intent and
the quality of outputs it produces. Useful strategies:
- Remove redundant instructions
- Use precise language instead of verbose explanation
- Eliminate preamble that doesn't change model behavior
- Consolidate repeated concepts
- Remove politeness that doesn't affect output quality

Target: 20-35% fewer tokens. Do not change what the prompt is asking for.
Do not remove constraints or safety instructions.

Return only the optimized prompt, nothing else."""

_QUALITY_EVAL_PROMPT = """\
Compare these two responses to the same task. Rate whether Response B maintains
the quality of Response A.

TASK: {task}

RESPONSE A (baseline):
{response_a}

RESPONSE B (candidate):
{response_b}

Does Response B meet or exceed the quality of Response A for this task?
Consider accuracy, completeness, relevance, and clarity.

Return JSON only:
{{"quality_maintained": true/false, "score": 0.0-1.0, "reasoning": "one sentence"}}"""


class PromptOptimizer:
    """
    Iteratively compresses prompt templates by testing variants against a
    quality baseline. Only promotes a variant if it passes the quality threshold.
    """

    def __init__(
        self,
        client,
        quality_threshold: float = 0.85,
        min_reduction: float = 0.10,
    ):
        self._client = client
        self._quality_threshold = quality_threshold
        self._min_reduction = min_reduction
        self._registry: dict[str, PromptVariant] = {}
        self._total_tokens_saved = 0
        self._optimization_calls = 0

    def optimize(
        self,
        name: str,
        template: str,
        test_tasks: list[str],
        quality_fn: Optional[Callable] = None,
        iterations: int = 3,
    ) -> OptimizationResult:
        """
        Run the optimization loop on a prompt template.

        Args:
            name: Identifier for this template (used with ``get()``).
            template: The prompt template to optimize.
            test_tasks: Sample tasks used to evaluate variant quality.
            quality_fn: Custom scoring function. Falls back to Haiku evaluation.
            iterations: How many optimization rounds to run.

        Returns:
            OptimizationResult with the best template found and token stats.
        """
        original_tokens = len(template.split())
        current = template
        current_tokens = original_tokens
        variants_tested = 0

        for _ in range(iterations):
            variant = self._generate_variant(current)
            if not variant:
                continue

            variant_tokens = len(variant.split())
            reduction = (current_tokens - variant_tokens) / max(1, current_tokens)
            if reduction < self._min_reduction:
                continue

            sample = random.sample(test_tasks, min(3, len(test_tasks)))
            scores = [
                self._evaluate(task, current, variant, quality_fn)
                for task in sample
            ]
            variants_tested += len(scores)

            if scores and sum(scores) / len(scores) >= self._quality_threshold:
                self._total_tokens_saved += (
                    (current_tokens - variant_tokens) * len(test_tasks)
                )
                current = variant
                current_tokens = variant_tokens

        self._registry[name] = PromptVariant(
            template=current,
            token_count=current_tokens,
            quality_score=self._quality_threshold,
        )

        return OptimizationResult(
            original_template=template,
            best_template=current,
            original_tokens=original_tokens,
            best_tokens=current_tokens,
            quality_maintained=True,
            token_reduction=(original_tokens - current_tokens) / max(1, original_tokens),
            variants_tested=variants_tested,
        )

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Return the current optimized template for a named prompt."""
        v = self._registry.get(name)
        return v.template if v else default

    def register(self, name: str, template: str) -> None:
        """Register a template without optimization."""
        self._registry[name] = PromptVariant(
            template=template,
            token_count=len(template.split()),
            quality_score=1.0,
        )

    def _generate_variant(self, template: str) -> Optional[str]:
        try:
            response = self._client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=len(template.split()) + 100,
                messages=[{
                    "role": "user",
                    "content": _VARIATION_PROMPT.format(template=template),
                }],
            )
            self._optimization_calls += 1
            return response.content[0].text.strip()
        except Exception:
            return None

    def _evaluate(
        self,
        task: str,
        baseline: str,
        candidate: str,
        quality_fn: Optional[Callable],
    ) -> float:
        if quality_fn:
            return quality_fn(task, baseline, candidate)

        try:
            r_a = self._client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=500,
                messages=[{"role": "user", "content": f"{baseline}\n\nTask: {task}"}],
            ).content[0].text

            r_b = self._client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=500,
                messages=[{"role": "user", "content": f"{candidate}\n\nTask: {task}"}],
            ).content[0].text

            eval_resp = self._client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=150,
                messages=[{
                    "role": "user",
                    "content": _QUALITY_EVAL_PROMPT.format(
                        task=task, response_a=r_a, response_b=r_b
                    ),
                }],
            ).content[0].text

            self._optimization_calls += 3
            return float(json.loads(eval_resp.strip()).get("score", 0.0))
        except Exception:
            return 0.0

    def stats(self) -> dict:
        avg_reduction = 0.0
        if self._registry:
            avg_reduction = sum(
                1 - v.token_count / max(1, len(v.template.split()))
                for v in self._registry.values()
            ) / len(self._registry)

        return {
            "templates_registered": len(self._registry),
            "optimization_calls": self._optimization_calls,
            "total_tokens_saved": self._total_tokens_saved,
            "avg_reduction": avg_reduction,
        }
