"""Tests for the main Conductor interface."""

from unittest.mock import MagicMock, patch

from conductor.core import Conductor, SessionStats, _build_prompt
from conductor.router import Tier


def _mock_client(text="result", input_tokens=100, output_tokens=50):
    client = MagicMock()
    resp = MagicMock()
    resp.content[0].text = text
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    client.messages.create.return_value = resp
    return client


class TestBuildPrompt:
    def test_no_context(self):
        assert _build_prompt("do this", "") == "do this"

    def test_with_context(self):
        result = _build_prompt("do this", "some background")
        assert "Context:" in result
        assert "Task:" in result
        assert "do this" in result
        assert "some background" in result


class TestSessionStats:
    def test_savings_pct_zero_baseline(self):
        s = SessionStats()
        assert s.savings_pct == 0.0

    def test_cost_vs_all_opus(self):
        s = SessionStats(total_tokens=1_000_000)
        # 1M tokens at $75/1M = $75
        assert s.cost_vs_all_opus == 75.0


class TestConductor:
    def test_basic_run(self):
        client = _mock_client("analysis complete")
        c = Conductor(client=client)

        with patch("conductor.core.TaskRouter") as MockRouter:
            decision = MagicMock()
            decision.tier = Tier.SONNET
            MockRouter.return_value.route.return_value = decision

            result = c.run("analyze this data")

        assert result is not None

    def test_stats_start_empty(self):
        c = Conductor(client=_mock_client())
        s = c.stats()
        assert s.total_calls == 0
        assert s.total_cost_usd == 0.0

    def test_route_only_does_not_execute(self):
        client = _mock_client()
        c = Conductor(client=client)

        with patch.object(c._router, "route") as mock_route:
            decision = MagicMock()
            decision.tier = Tier.HAIKU
            mock_route.return_value = decision
            result = c.route_only("classify this")

        # route_only should call router.route but not client.messages.create
        # (beyond what the router itself calls, which is mocked)
        mock_route.assert_called_once()
        assert result == decision

    def test_force_tier_respected(self):
        client = _mock_client()
        c = Conductor(client=client)
        result = c.run("do something simple", force_tier=Tier.HAIKU)
        assert result.tier == Tier.HAIKU

    def test_local_tier_falls_back_without_local_executor(self):
        client = _mock_client()
        c = Conductor(client=client, local_model=None)
        # forcing LOCAL without a local executor should fall back to frontier
        result = c.run("format this json", force_tier=Tier.LOCAL)
        assert result.tier == Tier.SONNET

    def test_print_stats_runs(self, capsys):
        client = _mock_client()
        c = Conductor(client=client)
        c.print_stats()
        out = capsys.readouterr().out
        assert "Conductor Session Stats" in out
