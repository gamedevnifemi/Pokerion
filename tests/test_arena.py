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
