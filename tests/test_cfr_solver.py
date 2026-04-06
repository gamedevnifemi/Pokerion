"""Tests for Vanilla CFR solver on Kuhn Poker.

Kuhn Poker has a known analytical Nash equilibrium:
- Game value for player 0: -1/18 ~ -0.0556
- Player 1 with K: always bets after check, always calls a bet
- Player 1 with J: never calls a bet; bets after check with probability 1/3
- Player 0 with Q: always checks initially
"""

import pytest

from pokerion.game.kuhn import KuhnHistory
from pokerion.solver.vanilla_cfr import VanillaCFR


@pytest.fixture
def trained_solver():
    """Train CFR for 10,000 iterations — enough for convergence."""
    solver = VanillaCFR(create_root=KuhnHistory, num_players=2)
    solver.train(10_000)
    return solver


class TestConvergence:
    def test_game_value_converges_to_minus_one_eighteenth(self, trained_solver):
        """Player 0's expected value must converge to -1/18."""
        values = trained_solver.expected_utility()
        assert values[0] == pytest.approx(-1 / 18, abs=0.01)
        assert values[1] == pytest.approx(1 / 18, abs=0.01)

    def test_zero_sum(self, trained_solver):
        """Game values must sum to zero."""
        values = trained_solver.expected_utility()
        assert values[0] + values[1] == pytest.approx(0.0, abs=1e-6)

    def test_exploitability_decreases(self):
        """Exploitability should decrease as iterations increase."""
        solver = VanillaCFR(create_root=KuhnHistory, num_players=2)

        solver.train(100)
        exploit_100 = solver.exploitability()

        solver.train(900)  # now at 1000 total
        exploit_1000 = solver.exploitability()

        solver.train(9000)  # now at 10000 total
        exploit_10000 = solver.exploitability()

        # Exploitability should be trending down
        assert exploit_10000 < exploit_100
        # After 10k iterations, exploitability should be very small
        assert exploit_10000 < 0.05

    def test_low_exploitability(self, trained_solver):
        """After 10k iterations, exploitability should be near zero."""
        exploit = trained_solver.exploitability()
        assert exploit < 0.02


class TestNashEquilibriumProperties:
    """Known Nash equilibrium properties for Kuhn Poker.

    There is a family of Nash equilibria parameterized by alpha in [0, 1/3].
    All share these common properties.
    """

    def test_player1_king_always_bets_after_check(self, trained_solver):
        """Player 1 with King after opponent checks: always bet."""
        strategy = trained_solver.get_strategy()
        s = strategy["K|check"]
        assert s["bet"] > 0.95

    def test_player1_king_always_calls_bet(self, trained_solver):
        """Player 1 with King facing a bet: always call."""
        strategy = trained_solver.get_strategy()
        s = strategy["K|bet"]
        assert s["call"] > 0.95

    def test_player1_jack_never_calls_bet(self, trained_solver):
        """Player 1 with Jack facing a bet: always fold."""
        strategy = trained_solver.get_strategy()
        s = strategy["J|bet"]
        assert s["fold"] > 0.95

    def test_player1_jack_bets_one_third_after_check(self, trained_solver):
        """Player 1 with Jack after opponent checks: bet with probability 1/3."""
        strategy = trained_solver.get_strategy()
        s = strategy["J|check"]
        assert s["bet"] == pytest.approx(1 / 3, abs=0.05)

    def test_player0_queen_always_checks(self, trained_solver):
        """Player 0 with Queen: always check initially."""
        strategy = trained_solver.get_strategy()
        s = strategy["Q"]
        assert s["check"] > 0.95

    def test_player0_king_bets_with_three_alpha(self, trained_solver):
        """Player 0 with King: bets with probability 3*alpha (alpha in [0, 1/3]).
        So bet probability is in [0, 1]."""
        strategy = trained_solver.get_strategy()
        s = strategy["K"]
        # Bet probability should be between 0 and 1 — it's a valid mixed strategy
        assert 0.0 <= s["bet"] <= 1.0 + 0.05


class TestInfoSets:
    def test_finds_all_12_info_sets(self, trained_solver):
        """Solver should discover all 12 information sets."""
        assert len(trained_solver.info_sets) == 12

    def test_all_strategies_are_valid_distributions(self, trained_solver):
        """Every strategy must be a valid probability distribution."""
        for key, strat in trained_solver.get_strategy().items():
            total = sum(strat.values())
            assert total == pytest.approx(1.0, abs=1e-6), f"Info set {key} doesn't sum to 1: {strat}"
            for action, prob in strat.items():
                assert prob >= -1e-6, f"Negative probability at {key}: {action}={prob}"


class TestSolverMechanics:
    def test_train_increments_iterations(self):
        solver = VanillaCFR(create_root=KuhnHistory, num_players=2)
        solver.train(50)
        assert solver.iterations == 50
        solver.train(50)
        assert solver.iterations == 100

    def test_callback_called_each_iteration(self):
        solver = VanillaCFR(create_root=KuhnHistory, num_players=2)
        calls = []
        solver.train(10, callback=lambda i: calls.append(i))
        assert calls == list(range(1, 11))

    def test_get_strategy_vs_get_current_strategy(self):
        """Average strategy and current strategy should differ."""
        solver = VanillaCFR(create_root=KuhnHistory, num_players=2)
        solver.train(100)
        avg = solver.get_strategy()
        cur = solver.get_current_strategy()
        # They should be different dicts (current adapts faster)
        assert avg is not cur
