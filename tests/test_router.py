"""Tests for the task routing layer."""

from unittest.mock import MagicMock

from conductor.router import TaskProfile, TaskRouter, Tier, _task_key


class TestTier:
    def test_values(self):
        assert Tier.HAIKU == "haiku"
        assert Tier.SONNET == "sonnet"
        assert Tier.OPUS == "opus"
        assert Tier.LOCAL == "local"

    def test_is_string(self):
        assert isinstance(Tier.HAIKU, str)


class TestTaskKey:
    def test_normalizes_case(self):
        assert _task_key("Analyze Revenue Trends") == _task_key("analyze revenue trends")

    def test_truncates_to_eight_words(self):
        long = "one two three four five six seven eight nine ten"
        key = _task_key(long)
        assert len(key.split()) == 8

    def test_consistent_for_same_prefix(self):
        # Both strings share the same first 8 words, so their keys should match
        a = _task_key("summarize the quarterly results for stakeholders in detail and more words")
        b = _task_key("summarize the quarterly results for stakeholders in detail but different suffix")
        assert a == b


class TestTaskProfile:
    def test_dominant_tier_requires_five_calls(self):
        p = TaskProfile(task_pattern="test")
        for _ in range(4):
            p.call_count += 1
            p.tier_counts["sonnet"] = p.tier_counts.get("sonnet", 0) + 1
        assert p.dominant_tier() is None

    def test_dominant_tier_requires_75_percent(self):
        p = TaskProfile(task_pattern="test", call_count=8)
        p.tier_counts = {"sonnet": 5, "haiku": 3}
        # 5/8 = 62.5% — below threshold
        assert p.dominant_tier() is None

    def test_dominant_tier_returns_when_threshold_met(self):
        p = TaskProfile(task_pattern="test", call_count=8)
        p.tier_counts = {"sonnet": 7, "haiku": 1}
        assert p.dominant_tier() == Tier.SONNET


class TestTaskRouter:
    def _make_response(self, tier="sonnet", score=0.6, confidence=0.8):
        resp = MagicMock()
        resp.content[0].text = (
            f'{{"tier": "{tier}", "score": {score}, '
            f'"reasoning": "test", "confidence": {confidence}}}'
        )
        resp.usage.input_tokens = 100
        resp.usage.output_tokens = 50
        return resp

    def test_force_tier_skips_api(self):
        client = MagicMock()
        router = TaskRouter(client)
        decision = router.route("anything", force_tier=Tier.OPUS)
        client.messages.create.assert_not_called()
        assert decision.tier == Tier.OPUS
        assert decision.confidence == 1.0

    def test_routes_via_haiku(self):
        client = MagicMock()
        client.messages.create.return_value = self._make_response("sonnet")
        router = TaskRouter(client)
        decision = router.route("draft a blog post about climate change")
        assert decision.tier == Tier.SONNET
        assert client.messages.create.called

    def test_fallback_to_sonnet_on_error(self):
        client = MagicMock()
        client.messages.create.side_effect = Exception("network error")
        router = TaskRouter(client)
        decision = router.route("some task")
        assert decision.tier == Tier.SONNET
        assert decision.confidence == 0.5

    def test_cache_hit_after_five_calls(self):
        client = MagicMock()
        client.messages.create.return_value = self._make_response("haiku")
        router = TaskRouter(client)

        task = "classify this support ticket"
        for _ in range(5):
            router.route(task)

        call_count_before = client.messages.create.call_count
        router.route(task)
        # 6th call should use cache, no additional API call
        assert client.messages.create.call_count == call_count_before

    def test_stats(self):
        client = MagicMock()
        client.messages.create.return_value = self._make_response()
        router = TaskRouter(client)
        router.route("test task")
        s = router.stats()
        assert s["routing_calls"] == 1
        assert "cache_hit_rate" in s
