"""Tests for the simple bots and for exact scoring of a table."""

import numpy as np
import pytest

from pokerion.arena import Entrant, best_reply_to_table, bots, evaluate_table, table_mix_curve
from pokerion.evaluation import best_response


def probability(tree, strategy, key, action):
    return strategy[tree.row(key), tree.column(action)]


class TestBots:
    @pytest.mark.parametrize(
        "make", [bots.uniform_random, bots.calling_station, bots.maniac, bots.rock]
    )
    def test_every_bot_plays_a_valid_strategy(self, kuhn3, make):
        kuhn3.check_strategy(make(kuhn3))

    def test_calling_station_never_bets_and_never_folds(self, kuhn):
        strategy = bots.calling_station(kuhn)
        assert probability(kuhn, strategy, "K", "check") == 1.0
        assert probability(kuhn, strategy, "J|bet", "call") == 1.0

    def test_maniac_bets_when_it_can_and_calls_when_it_cannot(self, kuhn):
        strategy = bots.maniac(kuhn)
        assert probability(kuhn, strategy, "J", "bet") == 1.0
        assert probability(kuhn, strategy, "J|bet", "call") == 1.0

    def test_rock_never_puts_in_a_chip(self, kuhn):
        strategy = bots.rock(kuhn)
        assert probability(kuhn, strategy, "K", "check") == 1.0
        assert probability(kuhn, strategy, "K|bet", "fold") == 1.0

    def test_prefer_takes_the_first_legal_preference(self, kuhn):
        strategy = bots.prefer(kuhn, "fold", "bet")
        assert probability(kuhn, strategy, "Q", "bet") == 1.0       # fold is not legal here
        assert probability(kuhn, strategy, "Q|bet", "fold") == 1.0

    def test_prefer_is_uniform_where_no_preference_is_legal(self, kuhn):
        strategy = bots.prefer(kuhn, "fold", "raise")  # Kuhn has no raise
        assert probability(kuhn, strategy, "Q", "check") == 0.5
        kuhn.check_strategy(strategy)

    def test_blend_mixes_at_every_information_set(self, kuhn):
        mixed = bots.blend(bots.rock(kuhn), bots.maniac(kuhn), 0.25)
        assert probability(kuhn, mixed, "K", "bet") == 0.25
        kuhn.check_strategy(mixed)

    def test_blend_endpoints(self, kuhn):
        rock, maniac = bots.rock(kuhn), bots.maniac(kuhn)
        assert np.array_equal(bots.blend(rock, maniac, 0.0), rock)
        assert np.array_equal(bots.blend(rock, maniac, 1.0), maniac)

    def test_blend_rejects_a_weight_outside_zero_to_one(self, kuhn):
        with pytest.raises(ValueError, match="between 0 and 1"):
            bots.blend(bots.rock(kuhn), bots.maniac(kuhn), 1.5)

    def test_random_strategy_is_valid_and_different_each_time(self, kuhn3):
        rng = np.random.default_rng(0)
        first, second = bots.random_strategy(kuhn3, rng), bots.random_strategy(kuhn3, rng)
        kuhn3.check_strategy(first)
        assert not np.allclose(first, second)

    def test_random_strategy_spreads_evenly_between_two_actions(self, kuhn):
        rng = np.random.default_rng(1)
        bet = [probability(kuhn, bots.random_strategy(kuhn, rng), "K", "bet") for _ in range(2000)]
        # Uniform on [0, 1]: mean one half, a quarter of the draws below one quarter.
        assert np.mean(bet) == pytest.approx(0.5, abs=0.03)
        assert np.mean(np.array(bet) < 0.25) == pytest.approx(0.25, abs=0.03)

    def test_perturbed_stays_within_the_spread(self, kuhn, kuhn_equilibrium):
        rng = np.random.default_rng(2)
        for _ in range(20):
            moved = bots.perturbed(kuhn, kuhn_equilibrium, 0.2, rng)
            kuhn.check_strategy(moved)
            assert np.abs(moved - kuhn_equilibrium).max() <= 0.2 + 1e-12

    def test_perturbed_actually_moves_the_strategy(self, kuhn, kuhn_equilibrium):
        moved = bots.perturbed(kuhn, kuhn_equilibrium, 0.2, np.random.default_rng(3))
        assert np.abs(moved - kuhn_equilibrium).max() > 0.05

    def test_perturbed_spreads_an_action_that_is_never_played(self, kuhn):
        # A rock never bets. Within 0.2 of never is anywhere from 0 to 20% of the time.
        rng = np.random.default_rng(6)
        rock = bots.rock(kuhn)
        bets = [
            probability(kuhn, bots.perturbed(kuhn, rock, 0.2, rng), "K", "bet") for _ in range(2000)
        ]
        assert min(bets) >= 0.0 and max(bets) <= 0.2
        assert np.mean(bets) == pytest.approx(0.1, abs=0.01)

    def test_perturbed_with_no_spread_changes_nothing(self, kuhn, kuhn_equilibrium):
        same = bots.perturbed(kuhn, kuhn_equilibrium, 0.0, np.random.default_rng(4))
        assert np.allclose(same, kuhn_equilibrium)

    def test_perturbed_rejects_a_negative_spread(self, kuhn):
        with pytest.raises(ValueError, match="spread"):
            bots.perturbed(kuhn, kuhn.uniform_strategy(), -0.1, np.random.default_rng(5))


class TestEvaluateTable:
    def test_winnings_sum_to_zero(self, kuhn3, kuhn3_equilibrium):
        table = [
            Entrant("solid", kuhn3_equilibrium),
            Entrant("rock", bots.rock(kuhn3)),
            Entrant("maniac", bots.maniac(kuhn3)),
        ]
        result = evaluate_table(kuhn3, table)
        assert result.names == ("solid", "rock", "maniac")
        assert result.winnings.sum() == pytest.approx(0.0, abs=1e-12)

    def test_identical_entrants_break_even(self, kuhn3):
        table = [Entrant("random", bots.uniform_random(kuhn3))] * 3
        assert evaluate_table(kuhn3, table).winnings == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)

    def test_by_seat_matches_playing_each_seating_directly(self, kuhn):
        rock, maniac = bots.rock(kuhn), bots.maniac(kuhn)
        result = evaluate_table(kuhn, [Entrant("rock", rock), Entrant("maniac", maniac)])
        rock_first = kuhn.expected_utilities(kuhn.combine([rock, maniac]))
        maniac_first = kuhn.expected_utilities(kuhn.combine([maniac, rock]))
        assert result.by_seat[0] == pytest.approx([rock_first[0], maniac_first[1]])
        assert result.by_seat[1] == pytest.approx([maniac_first[0], rock_first[1]])

    def test_solid_play_at_its_own_table_earns_the_equilibrium_payoff_in_each_seat(
        self, kuhn, kuhn_equilibrium
    ):
        result = evaluate_table(kuhn, [Entrant("solid", kuhn_equilibrium)] * 2)
        assert result.by_seat[0] == pytest.approx([-1 / 18, 1 / 18], abs=1e-4)

    @pytest.mark.parametrize(
        "make", [bots.uniform_random, bots.calling_station, bots.maniac, bots.rock]
    )
    def test_solid_play_profits_from_every_weak_bot(self, kuhn, kuhn_equilibrium, make):
        table = [Entrant("solid", kuhn_equilibrium), Entrant("weak", make(kuhn))]
        assert evaluate_table(kuhn, table).winnings[0] > 0.05

    def test_needs_one_entrant_per_seat(self, kuhn3):
        with pytest.raises(ValueError, match="seats 3 players"):
            evaluate_table(kuhn3, [Entrant("random", bots.uniform_random(kuhn3))] * 2)

    def test_rejects_an_invalid_strategy(self, kuhn):
        broken = Entrant("broken", np.zeros((kuhn.num_infosets, kuhn.num_actions)))
        with pytest.raises(ValueError, match="sum to 1"):
            evaluate_table(kuhn, [broken, broken])


class TestBestReplyToTable:
    def test_a_rock_can_be_robbed_of_its_ante_every_hand(self, kuhn, kuhn_equilibrium):
        # Bet every hand and the rock folds every hand.
        table = [Entrant("solid", kuhn_equilibrium), Entrant("rock", bots.rock(kuhn))]
        assert best_reply_to_table(kuhn, table, hero=0) == pytest.approx(1.0)

    def test_ceiling_against_random_play_is_half_the_published_nash_conv(self, kuhn):
        # NashConv of uniform play is 11/12, shared between the two seats.
        table = [Entrant("anyone", bots.rock(kuhn)), Entrant("random", bots.uniform_random(kuhn))]
        assert best_reply_to_table(kuhn, table, hero=0) == pytest.approx(11 / 24)

    def test_ceiling_is_at_least_what_the_hero_wins(self, kuhn3, kuhn3_equilibrium):
        table = [
            Entrant("solid", kuhn3_equilibrium),
            Entrant("station", bots.calling_station(kuhn3)),
            Entrant("random", bots.uniform_random(kuhn3)),
        ]
        won = evaluate_table(kuhn3, table).winnings[0]
        assert best_reply_to_table(kuhn3, table, hero=0) >= won

    def test_ceiling_is_reached_by_the_best_response_itself(self, kuhn):
        maniac = bots.maniac(kuhn)
        table = [Entrant("anyone", bots.rock(kuhn)), Entrant("maniac", maniac)]
        ceiling = best_reply_to_table(kuhn, table, hero=0)
        # Seat the best responses by hand and average the two seats.
        first = best_response(kuhn, kuhn.combine([maniac, maniac]), 0).value
        second = best_response(kuhn, kuhn.combine([maniac, maniac]), 1).value
        assert ceiling == pytest.approx((first + second) / 2)


class TestTableMixCurve:
    def test_one_point_per_number_of_weak_opponents(self, kuhn3, kuhn3_equilibrium):
        solid = Entrant("solid", kuhn3_equilibrium)
        weak = Entrant("random", bots.uniform_random(kuhn3))
        curve = table_mix_curve(kuhn3, hero=solid, strong=solid, weak=weak)
        assert curve.shape == (3,)
        assert curve[0] == pytest.approx(0.0, abs=1e-12)  # a table of itself breaks even
        assert curve[2] == pytest.approx(
            evaluate_table(kuhn3, [solid, weak, weak]).winnings[0]
        )

    @pytest.mark.parametrize(
        "make", [bots.uniform_random, bots.calling_station, bots.maniac, bots.rock]
    )
    def test_solid_play_wins_more_as_weak_bots_join(self, kuhn3, kuhn3_equilibrium, make):
        solid = Entrant("solid", kuhn3_equilibrium)
        curve = table_mix_curve(kuhn3, solid, solid, Entrant("weak", make(kuhn3)))
        assert curve[0] < curve[1] < curve[2]
