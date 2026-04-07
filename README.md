# conductor

Tiered AI orchestration for the Anthropic API. Routes each task to the cheapest
model that can handle it — Haiku for simple classification, Sonnet for research
and drafting, Opus for genuinely ambiguous high-stakes decisions.

```
pip install conductor-ai
```

---

## How it works

**Task routing** — A cheap Haiku call scores each task and picks the right tier.
Haiku handles routing and classification (~60% of calls, ~5% of cost). Sonnet
handles daily workload (~30% of calls, ~40% of cost). Opus only gets called when
nuance on ambiguous evidence actually matters (~10% of calls, ~55% of cost).

Routing accuracy improves as the system accumulates history — repeated task
patterns get cached decisions with no scoring call at all.

**Context compression** — Before a Sonnet or Opus call, Haiku builds a focused
brief from your full context. Input tokens drop 60–70%. The expensive model never
sees what it doesn't need.

**Prompt optimization** — An automated loop tests prompt variants against a
quality baseline and keeps whichever produces equivalent output with fewer tokens.

**Local execution** — Ollama and LM Studio are supported. Execution tasks route
to local hardware at $0/call with automatic fallback to Sonnet if unavailable.

---

## Quick start

```python
import anthropic
from conductor import Conductor

client = anthropic.Anthropic()
c = Conductor(client=client)

result = c.run("Analyze Q3 revenue trends and flag anomalies")
print(result.content)
print(f"tier={result.tier.value}  cost=${result.cost_usd:.4f}")
```

## With context compression

```python
with open("knowledge_base.txt") as f:
    context = f.read()

result = c.run("Identify the three highest-risk items", context=context)
print(f"Saved {result.brief.savings:,} tokens ({result.brief.savings_pct:.0f}%)")
```

## Force a specific tier

```python
from conductor import Tier

result = c.run(
    "Critical legal analysis with ambiguous precedent",
    force_tier=Tier.OPUS,
)

result = c.run("Format this JSON", force_tier=Tier.LOCAL)
```

## With a local model

```python
# Requires Ollama or LM Studio running locally
c = Conductor(client=client, local_model="qwen3:32b")
```

Install [Ollama](https://ollama.ai) and pull a model:

```bash
ollama pull qwen3:32b
ollama pull gemma3:27b
ollama pull llama3.3:70b
```

[LM Studio](https://lmstudio.ai) is also supported — conductor detects it on port 1234.

## Prompt optimization

```python
from conductor import PromptOptimizer

optimizer = PromptOptimizer(client)
result = optimizer.optimize(
    name="analysis_prompt",
    template=your_existing_prompt,
    test_tasks=["sample task 1", "sample task 2"],
    iterations=3,
)
print(f"Token reduction: {result.token_reduction * 100:.0f}%")

# Use the optimized version
optimized = optimizer.get("analysis_prompt")
```

## Session stats

```python
c.print_stats()

# Conductor Session Stats
#   Total calls:       247
#   Tier distribution: {'haiku': 148, 'sonnet': 74, 'opus': 25}
#   Total tokens:      187,432
#   Total cost:        $1.24
#   vs all-Opus:       $14.06
#   Savings:           91%
#   Context compression saved: 43,218 tokens
#   Router cache hits: 31 (21%)
```

---

## Requirements

- Python 3.10+
- `anthropic` SDK
- Ollama or LM Studio (optional, for local execution)

## License

MIT
