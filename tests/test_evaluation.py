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
