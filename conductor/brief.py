from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Brief:
    content: str
    source_tokens: int
    brief_tokens: int
    compression_ratio: float
    relevance_score: float

    @property
    def savings(self) -> int:
        return self.source_tokens - self.brief_tokens

    @property
    def savings_pct(self) -> float:
        if self.source_tokens == 0:
            return 0.0
        return (self.savings / self.source_tokens) * 100


_BRIEF_PROMPT = """\
You are a context compression specialist.

TASK TO ACCOMPLISH:
{task}

FULL CONTEXT:
{context}

Extract only what is directly relevant to the task above. Be selective — omit
anything the model handling this task doesn't need.

Return a focused brief. No preamble, no explanation. Just the relevant content,
reorganized for clarity. Target: 30% of the original length or less."""


class BriefBuilder:
    """
    Compresses large context documents down to a focused brief using Haiku.

    Skips compression when the context is already within the token budget.
    Falls back to a truncated excerpt on API errors rather than raising.
    """

    def __init__(self, client):
        self._client = client
        self._total_source_tokens = 0
        self._total_brief_tokens = 0
        self._calls = 0

    def build(
        self,
        task: str,
        context: str,
        max_brief_tokens: int = 2000,
    ) -> Brief:
        if not context or not context.strip():
            return Brief(
                content="",
                source_tokens=0,
                brief_tokens=0,
                compression_ratio=1.0,
                relevance_score=1.0,
            )

        # rough estimate: ~4 chars per token
        est_source = len(context) // 4
        if est_source <= max_brief_tokens:
            return Brief(
                content=context,
                source_tokens=est_source,
                brief_tokens=est_source,
                compression_ratio=1.0,
                relevance_score=1.0,
            )

        prompt = _BRIEF_PROMPT.format(task=task[:300], context=context[:12000])

        try:
            response = self._client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=max_brief_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            self._calls += 1
            brief_content = response.content[0].text.strip()
            brief_tokens = response.usage.output_tokens
            source_tokens = response.usage.input_tokens

            self._total_source_tokens += source_tokens
            self._total_brief_tokens += brief_tokens

            return Brief(
                content=brief_content,
                source_tokens=source_tokens,
                brief_tokens=brief_tokens,
                compression_ratio=source_tokens / max(1, brief_tokens),
                relevance_score=0.9,
            )

        except Exception:
            truncated = context[: max_brief_tokens * 4]
            est = len(truncated) // 4
            return Brief(
                content=truncated,
                source_tokens=est_source,
                brief_tokens=est,
                compression_ratio=est_source / max(1, est),
                relevance_score=0.5,
            )

    def build_multi(
        self,
        task: str,
        sources: list[str],
        max_total_tokens: int = 3000,
    ) -> Brief:
        """Compress multiple context sources, splitting the token budget evenly."""
        if not sources:
            return Brief("", 0, 0, 1.0, 1.0)

        per_source = max_total_tokens // len(sources)
        parts: list[str] = []
        total_source = 0
        total_brief = 0

        for source in sources:
            b = self.build(task, source, per_source)
            parts.append(b.content)
            total_source += b.source_tokens
            total_brief += b.brief_tokens

        combined = "\n\n---\n\n".join(p for p in parts if p)
        return Brief(
            content=combined,
            source_tokens=total_source,
            brief_tokens=total_brief,
            compression_ratio=total_source / max(1, total_brief),
            relevance_score=0.85,
        )

    def stats(self) -> dict:
        saved = self._total_source_tokens - self._total_brief_tokens
        ratio = 1 - self._total_brief_tokens / max(1, self._total_source_tokens)
        return {
            "calls": self._calls,
            "total_source_tokens": self._total_source_tokens,
            "total_brief_tokens": self._total_brief_tokens,
            "tokens_saved": saved,
            "compression_pct": ratio * 100,
        }
