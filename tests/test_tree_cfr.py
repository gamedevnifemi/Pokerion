"""Tests for CFR on the compiled tree, checked against published results.

Two-player Kuhn: the first player's equilibrium value is -1/18, and every
equilibrium shares the properties tested below (Kuhn 1950).

Three-player Kuhn: Szafron, Gibson and Sturtevant (AAMAS 2013), Table 3, give a
family of equilibria in which the payoffs are

    P1: -(1/2 + b) / 24      P2: -1/48      P3: (1 + b) / 24

for a parameter b between 0 and 1/4. So the second player always gets -1/48,
the first gets between -1/32 and -1/48, and the third between 1/24 and 5/96.
"""

import numpy as np
import pytest

from pokerion.evaluation import deviation_gains, nash_conv
from pokerion.game.tree import GameTree
from pokerion.solver.regret_models import RegretTable
from pokerion.solver.tree_cfr import TreeCFR, regret_matching
from pokerion.solver.vanilla_cfr import VanillaCFR
from tests.helpers import BiasedCoin


class TestRegretMatching:
    legal = np.array([[True, True, False], [True, True, True]])

    def test_plays_in_proportion_to_positive_regret(self):
        strategy = regret_matching(np.array([[3.0, 1.0, 0.0], [0.0, 2.0, 2.0]]), self.legal)
        assert strategy == pytest.approx(np.array([[0.75, 0.25, 0.0], [0.0, 0.5, 0.5]]))

    def test_ignores_negative_regret(self):
        strategy = regret_matching(np.array([[-5.0, 1.0, 0.0], [1.0, -1.0, 1.0]]), self.legal)
        assert strategy == pytest.approx(np.array([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5]]))

    def test_uniform_over_legal_actions_when_nothing_is_positive(self):
        strategy = regret_matching(np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -2.0]]), self.legal)
        assert strategy == pytest.approx(np.array([[0.5, 0.5, 0.0], [1 / 3, 1 / 3, 1 / 3]]))

    def test_never_plays_an_illegal_action(self):
        strategy = regret_matching(np.array([[0.0, 0.0, 9.0], [1.0, 1.0, 1.0]]), self.legal)
        assert strategy[0, 2] == 0.0


class TestRegretTable:
    def test_plain_table_keeps_the_running_total(self, kuhn):
        table = RegretTable(kuhn)
        step = np.full((kuhn.num_infosets, kuhn.num_actions), -1.0)
        table.update(step)
        table.update(step)
        assert (table.estimates() == -2.0).all()

    def test_floored_table_never_goes_below_zero(self, kuhn):
        table = RegretTable(kuhn, floored=True)
        table.update(np.full((kuhn.num_infosets, kuhn.num_actions), -1.0))
        assert (table.estimates() == 0.0).all()
        table.update(np.full((kuhn.num_infosets, kuhn.num_actions), 0.5))
        assert (table.estimates() == 0.5).all()


class TestSolverMechanics:
    def test_counts_iterations_across_calls(self, kuhn):
        solver = TreeCFR(kuhn)
        solver.train(5)
        solver.train(5)
        assert solver.iterations == 10

    def test_callback_runs_after_each_iteration(self, kuhn):
        calls = []
        TreeCFR(kuhn).train(4, callback=calls.append)
        assert calls == [1, 2, 3, 4]

    def test_strategies_are_valid_from_the_start(self, kuhn3):
        solver = TreeCFR(kuhn3)
        kuhn3.check_strategy(solver.current_strategy())
        kuhn3.check_strategy(solver.average_strategy())
        solver.train(3)
        kuhn3.check_strategy(solver.current_strategy())
        kuhn3.check_strategy(solver.average_strategy())

    def test_uses_the_regret_model_it_is_given(self, kuhn):
        class CountingTable(RegretTable):
            updates = 0

            def update(self, regrets):
                type(self).updates += 1
                super().update(regrets)

        TreeCFR(kuhn, CountingTable(kuhn)).train(3)
        assert CountingTable.updates == 3 * 2  # alternating: one update per seat per iteration

    def test_simultaneous_updates_touch_the_model_once_per_iteration(self, kuhn):
        seen = []

        class RecordingTable(RegretTable):
            def update(self, regrets):
                seen.append(np.count_nonzero(np.abs(regrets).sum(axis=1)))
                super().update(regrets)

        TreeCFR(kuhn, RecordingTable(kuhn), alternating=False).train(1)
        assert seen == [kuhn.num_infosets]  # both seats' rows arrive together


class TestTwoPlayerKuhn:
    @pytest.mark.parametrize("alternating", [True, False])
    def test_plain_cfr_reaches_the_game_value(self, kuhn, alternating):
        solver = TreeCFR(kuhn, alternating=alternating)
        solver.train(2000)
        average = solver.average_strategy()
        assert kuhn.expected_utilities(average)[0] == pytest.approx(-1 / 18, abs=2e-3)
        assert nash_conv(kuhn, average) < 0.02

    def test_gets_closer_with_more_iterations(self, kuhn):
        solver = TreeCFR(kuhn)
        solver.train(50)
        early = nash_conv(kuhn, solver.average_strategy())
        solver.train(1950)
        assert nash_conv(kuhn, solver.average_strategy()) < early / 5

    def test_cfr_plus_reaches_equilibrium(self, kuhn, kuhn_equilibrium):
        assert kuhn.expected_utilities(kuhn_equilibrium) == pytest.approx(
            [-1 / 18, 1 / 18], abs=1e-4
        )
        assert nash_conv(kuhn, kuhn_equilibrium) < 1e-3

    def test_equilibrium_has_the_known_properties(self, kuhn, kuhn_equilibrium):
        strategy = kuhn.strategy_to_dict(kuhn_equilibrium)
        assert strategy["K|check"]["bet"] > 0.99       # second player always bets a king
        assert strategy["K|bet"]["call"] > 0.99        # and always calls with it
        assert strategy["J|bet"]["fold"] > 0.99        # never calls with a jack
        assert strategy["J|check"]["bet"] == pytest.approx(1 / 3, abs=0.02)  # bluffs a third
        assert strategy["Q"]["check"] > 0.99           # first player never bets a queen
        # The first player bluffs a jack one third as often as it value-bets a king.
        assert strategy["K"]["bet"] == pytest.approx(3 * strategy["J"]["bet"], abs=0.02)

    def test_current_strategy_differs_from_the_average(self, kuhn):
        solver = TreeCFR(kuhn)
        solver.train(100)
        assert not np.allclose(solver.current_strategy(), solver.average_strategy())


class TestThreePlayerKuhn:
    def test_finds_an_equilibrium(self, kuhn3, kuhn3_equilibrium):
        # Not guaranteed with three players, but it is what the literature reports here.
        assert (deviation_gains(kuhn3, kuhn3_equilibrium) < 1e-3).all()

    def test_second_player_gets_the_published_payoff(self, kuhn3, kuhn3_equilibrium):
        payoffs = kuhn3.expected_utilities(kuhn3_equilibrium)
        assert payoffs[1] == pytest.approx(-1 / 48, abs=5e-4)

    def test_other_payoffs_fall_in_the_published_ranges(self, kuhn3, kuhn3_equilibrium):
        first, _, third = kuhn3.expected_utilities(kuhn3_equilibrium)
        slack = 5e-4
        assert -1 / 32 - slack <= first <= -1 / 48 + slack
        assert 1 / 24 - slack <= third <= 5 / 96 + slack

    def test_payoffs_fit_the_published_formula(self, kuhn3, kuhn3_equilibrium):
        first, _, third = kuhn3.expected_utilities(kuhn3_equilibrium)
        b = -24 * first - 0.5  # solve P1's payoff for the family's parameter
        assert third == pytest.approx((1 + b) / 24, abs=1e-3)

    def test_plain_cfr_also_approaches_equilibrium(self, kuhn3):
        solver = TreeCFR(kuhn3)
        solver.train(1000)
        average = solver.average_strategy()
        assert (deviation_gains(kuhn3, average) < 0.01).all()
        assert kuhn3.expected_utilities(average)[1] == pytest.approx(-1 / 48, abs=2e-3)


class TestChanceWeighting:
    def test_learns_to_guess_the_likelier_side_of_a_biased_coin(self):
        # Regrets must be weighted by chance: unweighted, heads and tails look the same.
        tree = GameTree(BiasedCoin())
        solver = TreeCFR(tree)
        solver.train(200)
        average = solver.average_strategy()
        assert average[tree.row("guess"), tree.column("h")] > 0.99
        assert tree.expected_utilities(average)[0] == pytest.approx(0.4, abs=0.01)

    @pytest.mark.xfail(
        strict=True,
        reason="VanillaCFR does not weight regrets by chance probability, so it is only "
        "correct when chance is uniform. See 'Two solvers, for now' in docs/architecture.md.",
    )
    def test_recursive_solver_on_the_same_biased_coin(self):
        # A record of a known gap in the site's solver, which Kuhn's uniform deals hide.
        # When the gap is closed this test starts passing, and the marker must come off.
        solver = VanillaCFR(create_root=BiasedCoin, num_players=2)
        solver.train(200)
        assert solver.get_strategy()["guess"]["h"] > 0.99
