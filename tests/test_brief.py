"""Tests for context compression."""

from unittest.mock import MagicMock

from conductor.brief import Brief, BriefBuilder


class TestBrief:
    def test_savings(self):
        b = Brief(
            content="compressed",
            source_tokens=1000,
            brief_tokens=300,
            compression_ratio=3.3,
            relevance_score=0.9,
        )
        assert b.savings == 700

    def test_savings_pct(self):
        b = Brief("", 1000, 400, 2.5, 0.9)
        assert b.savings_pct == 60.0

    def test_savings_pct_zero_source(self):
        b = Brief("", 0, 0, 1.0, 1.0)
        assert b.savings_pct == 0.0


class TestBriefBuilder:
    def test_empty_context_returns_empty_brief(self):
        client = MagicMock()
        builder = BriefBuilder(client)
        result = builder.build("task", "")
        assert result.content == ""
        assert result.source_tokens == 0
        client.messages.create.assert_not_called()

    def test_small_context_skips_compression(self):
        client = MagicMock()
        builder = BriefBuilder(client)
        small = "short context"
        result = builder.build("task", small, max_brief_tokens=2000)
        assert result.content == small
        assert result.compression_ratio == 1.0
        client.messages.create.assert_not_called()

    def test_large_context_calls_api(self):
        client = MagicMock()
        resp = MagicMock()
        resp.content[0].text = "compressed brief"
        resp.usage.output_tokens = 80
        resp.usage.input_tokens = 600
        client.messages.create.return_value = resp

        builder = BriefBuilder(client)
        large = "x " * 10000  # ~5000 estimated tokens
        result = builder.build("task", large, max_brief_tokens=100)

        assert client.messages.create.called
        assert result.content == "compressed brief"
        assert result.savings > 0

    def test_api_error_returns_truncated_fallback(self):
        client = MagicMock()
        client.messages.create.side_effect = Exception("timeout")
        builder = BriefBuilder(client)
        large = "x " * 10000
        result = builder.build("task", large, max_brief_tokens=100)
        # should not raise, returns truncated content
        assert result.content != ""
        assert result.relevance_score == 0.5

    def test_build_multi_empty_sources(self):
        builder = BriefBuilder(MagicMock())
        result = builder.build_multi("task", [])
        assert result.content == ""

    def test_stats_accumulate(self):
        client = MagicMock()
        resp = MagicMock()
        resp.content[0].text = "brief"
        resp.usage.output_tokens = 100
        resp.usage.input_tokens = 500
        client.messages.create.return_value = resp

        builder = BriefBuilder(client)
        large = "x " * 10000
        builder.build("task", large, max_brief_tokens=50)

        s = builder.stats()
        assert s["calls"] == 1
        assert s["tokens_saved"] > 0
