"""Tests for the exact yardsticks: best response, deviation gains, NashConv."""

from itertools import product

import numpy as np
import pytest

from pokerion.evaluation import best_response, deviation_gains, nash_conv
from pokerion.game.kuhn import KUHN_2P
from pokerion.game.tree import GameTree
from pokerion.solver.vanilla_cfr import VanillaCFR
from tests.helpers import BiasedCoin, random_pure_strategy, random_strategy


def brute_force_best_value(tree: GameTree, profile: np.ndarray, player: int) -> float:
    """Try every deterministic strategy for `player` and keep the best payoff."""
    rows = np.flatnonzero(tree.infoset_player == player)
    options = [np.flatnonzero(tree.legal[row]) for row in rows]
    best = -np.inf
    for columns in product(*options):
        candidate = profile.copy()
        candidate[rows] = 0.0
        candidate[rows, list(columns)] = 1.0
        best = max(best, tree.expected_utilities(candidate)[player])
    return best


class TestBestResponse:
    def test_matches_trying_every_pure_strategy(self, kuhn):
        rng = np.random.default_rng(0)
        for _ in range(3):
            profile = random_strategy(kuhn, rng)
            for player in range(2):
                found = best_response(kuhn, profile, player).value
                assert found == pytest.approx(brute_force_best_value(kuhn, profile, player))

    def test_value_is_what_the_returned_profile_earns(self, kuhn3):
        profile = random_strategy(kuhn3, np.random.default_rng(1))
        for player in range(3):
            response = best_response(kuhn3, profile, player)
            earned = kuhn3.expected_utilities(response.profile)[player]
            assert response.value == pytest.approx(earned)

    def test_only_the_responding_seat_changes(self, kuhn3):
        profile = random_strategy(kuhn3, np.random.default_rng(2))
        response = best_response(kuhn3, profile, 1)
        others = kuhn3.infoset_player != 1
        assert np.array_equal(response.profile[others], profile[others])
        mine = response.profile[~others]
        assert set(np.unique(mine)) == {0.0, 1.0}  # deterministic
        kuhn3.check_strategy(response.profile)

    def test_beats_any_other_strategy_for_that_seat(self, kuhn3):
        rng = np.random.default_rng(3)
        profile = random_strategy(kuhn3, rng)
        rows = kuhn3.infoset_player == 2
        best = best_response(kuhn3, profile, 2).value
        for _ in range(200):
            rival = profile.copy()
            rival[rows] = random_pure_strategy(kuhn3, rng)[rows]
            assert kuhn3.expected_utilities(rival)[2] <= best + 1e-12

    def test_weights_nodes_by_chance(self):
        # Guessing heads on a 70% coin is worth 0.4; an unweighted method would see a tie.
        tree = GameTree(BiasedCoin())
        response = best_response(tree, tree.uniform_strategy(), 0)
        assert response.value == pytest.approx(0.4)
        assert response.profile[tree.row("guess"), tree.column("h")] == 1.0


class TestBestResponseTies:
    """Equal actions do not change the value, but the choice between them is still a choice."""

    def chosen(self, tree, response, key):
        return tree.actions[int(response.profile[tree.row(key)].argmax())]

    def never_bets(self, kuhn):
        passive = {key: {a: float(a in ("check", "fold")) for a in kuhn.infoset_actions[row]}
                   for row, key in enumerate(kuhn.infoset_keys)}
        return kuhn.strategy_from_dict(passive)

    def test_prefer_decides_where_the_player_is_never_reached(self, kuhn):
        # The opponent never bets, so the first player never faces a bet after checking.
        table = self.never_bets(kuhn)
        folds = best_response(kuhn, table, 0, prefer=("fold",))
        calls = best_response(kuhn, table, 0, prefer=("call",))
        assert self.chosen(kuhn, folds, "K|check:bet") == "fold"
        assert self.chosen(kuhn, calls, "K|check:bet") == "call"
        assert folds.value == pytest.approx(calls.value)

    def test_prefer_does_not_override_a_clearly_better_action(self, kuhn):
        # Against an opponent that folds to every bet, betting a jack wins and checking it loses.
        response = best_response(kuhn, self.never_bets(kuhn), 0, prefer=("check",))
        assert self.chosen(kuhn, response, "J") == "bet"

    def test_tolerance_lets_prefer_choose_among_near_equals(self, kuhn, kuhn_equilibrium):
        # At equilibrium the first player is indifferent between betting and checking a king.
        bets = best_response(kuhn, kuhn_equilibrium, 0, prefer=("bet",), tolerance=1e-3)
        checks = best_response(kuhn, kuhn_equilibrium, 0, prefer=("check",), tolerance=1e-3)
        assert self.chosen(kuhn, bets, "K") == "bet"
        assert self.chosen(kuhn, checks, "K") == "check"
        assert bets.value == pytest.approx(checks.value, abs=1e-2)

    def test_value_is_still_what_the_chosen_response_earns(self, kuhn3):
        profile = random_strategy(kuhn3, np.random.default_rng(6))
        response = best_response(kuhn3, profile, 1, prefer=("fold", "check"), tolerance=0.05)
        exact = best_response(kuhn3, profile, 1)
        assert response.value == pytest.approx(kuhn3.expected_utilities(response.profile)[1])
        assert exact.value - 0.05 * 16 <= response.value <= exact.value + 1e-12

    def test_rejects_a_negative_tolerance(self, kuhn):
        with pytest.raises(ValueError, match="tolerance"):
            best_response(kuhn, kuhn.uniform_strategy(), 0, tolerance=-1.0)


class TestDeviationGains:
    def test_never_negative(self, kuhn3):
        rng = np.random.default_rng(4)
        for _ in range(5):
            assert (deviation_gains(kuhn3, random_strategy(kuhn3, rng)) >= -1e-12).all()

    def test_nash_conv_adds_the_gains(self, kuhn3):
        profile = random_strategy(kuhn3, np.random.default_rng(5))
        assert nash_conv(kuhn3, profile) == pytest.approx(deviation_gains(kuhn3, profile).sum())

    def test_uniform_play_in_kuhn_matches_the_published_value(self, kuhn):
        # 11/12: the value OpenSpiel's exploitability tests use for uniform play in Kuhn poker.
        assert nash_conv(kuhn, kuhn.uniform_strategy()) == pytest.approx(11 / 12)

    def test_agrees_with_the_recursive_solver(self, kuhn):
        # Two implementations of best response that share no code, on the same strategy.
        solver = VanillaCFR(create_root=KUHN_2P.root, num_players=2)
        solver.train(200)
        strategy = kuhn.strategy_from_dict(solver.get_strategy())
        assert nash_conv(kuhn, strategy) == pytest.approx(solver.exploitability(), abs=1e-9)
        assert kuhn.expected_utilities(strategy) == pytest.approx(solver.expected_utility())
