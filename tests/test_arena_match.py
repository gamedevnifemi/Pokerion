"""Tests for match play: scoring, seating, and what each agent is shown."""

import numpy as np
import pytest

from pokerion.arena import Entrant, bots, evaluate_table, play_match, play_matches


class Spy:
    """A fixed strategy that keeps everything the match tells it."""

    def __init__(self, name, strategy):
        self.name = name
        self.strategy = strategy
        self.me = None
        self.views = []

    def begin_match(self, tree, me):
        self.me = me
        self.views = []

    def strategy_for(self, seating):
        return self.strategy

    def observe(self, hand):
        self.views.append(hand)


def rng():
    return np.random.default_rng(0)


class TestScoring:
    def test_fixed_agents_score_the_same_as_a_static_table(self, kuhn3, kuhn3_equilibrium):
        table = [
            Entrant("solid", kuhn3_equilibrium),
            Entrant("rock", bots.rock(kuhn3)),
            Entrant("maniac", bots.maniac(kuhn3)),
        ]
        result = play_match(kuhn3, table, hands=12, rng=rng())  # twice round the 6 seatings
        assert result.names == ("solid", "rock", "maniac")
        assert result.winnings == pytest.approx(evaluate_table(kuhn3, table).winnings)

    def test_every_hand_is_zero_sum(self, kuhn3):
        table = [Entrant(name, make(kuhn3)) for name, make in
                 [("rock", bots.rock), ("maniac", bots.maniac), ("random", bots.uniform_random)]]
        result = play_match(kuhn3, table, hands=20, rng=rng())
        assert result.per_hand.shape == (3, 20)
        assert np.abs(result.per_hand.sum(axis=0)).max() < 1e-12

    def test_the_score_does_not_depend_on_the_cards_dealt(self, kuhn):
        table = [Entrant("rock", bots.rock(kuhn)), Entrant("random", bots.uniform_random(kuhn))]
        first = play_match(kuhn, table, hands=10, rng=np.random.default_rng(1))
        second = play_match(kuhn, table, hands=10, rng=np.random.default_rng(2))
        assert np.array_equal(first.per_hand, second.per_hand)


class TestSeating:
    def test_agents_take_turns_in_every_seat(self, kuhn):
        rock, maniac = bots.rock(kuhn), bots.maniac(kuhn)
        result = play_match(kuhn, [Entrant("rock", rock), Entrant("maniac", maniac)], 4, rng())
        rock_first = kuhn.expected_utilities(kuhn.combine([rock, maniac]))[0]
        rock_second = kuhn.expected_utilities(kuhn.combine([maniac, rock]))[1]
        assert result.per_hand[0] == pytest.approx([rock_first, rock_second] * 2)

    def test_a_single_seating_keeps_everyone_in_place(self, kuhn):
        rock, maniac = bots.rock(kuhn), bots.maniac(kuhn)
        result = play_match(
            kuhn, [Entrant("rock", rock), Entrant("maniac", maniac)], 3, rng(), seatings=[(1, 0)]
        )
        maniac_first = kuhn.expected_utilities(kuhn.combine([maniac, rock]))
        assert result.per_hand[1] == pytest.approx([maniac_first[0]] * 3)
        assert result.per_hand[0] == pytest.approx([maniac_first[1]] * 3)

    def test_each_agent_is_told_its_number_and_shown_every_hand(self, kuhn3):
        spies = [Spy(f"spy{i}", bots.uniform_random(kuhn3)) for i in range(3)]
        play_match(kuhn3, spies, hands=7, rng=rng())
        assert [spy.me for spy in spies] == [0, 1, 2]
        assert [len(spy.views) for spy in spies] == [7, 7, 7]
        for spy in spies:
            assert all(view.seating[view.seat] == spy.me for view in spy.views)

    def test_a_new_match_starts_from_scratch(self, kuhn):
        spies = [Spy("a", bots.rock(kuhn)), Spy("b", bots.maniac(kuhn))]
        play_match(kuhn, spies, hands=5, rng=rng())
        play_match(kuhn, spies, hands=2, rng=rng())
        assert [len(spy.views) for spy in spies] == [2, 2]


class TestWhatAgentsAreShown:
    def test_views_record_the_betting_in_order(self, kuhn):
        maniac, rock = Spy("maniac", bots.maniac(kuhn)), Spy("rock", bots.rock(kuhn))
        play_match(kuhn, [maniac, rock], hands=1, rng=rng(), seatings=[(0, 1)])
        view = maniac.views[0]
        assert [d.seat for d in view.decisions] == [0, 1]
        assert [kuhn.actions[d.action] for d in view.decisions] == ["bet", "fold"]

    def test_an_agent_always_knows_its_own_information_sets(self, kuhn3):
        spies = [Spy(f"spy{i}", bots.uniform_random(kuhn3)) for i in range(3)]
        play_match(kuhn3, spies, hands=30, rng=rng())
        for spy in spies:
            for view in spy.views:
                mine = [d for d in view.decisions if d.seat == view.seat]
                assert mine and all(d.row is not None for d in mine)
                assert all(kuhn3.infoset_player[d.row] == view.seat for d in mine)

    def test_a_fold_keeps_the_folded_cards_hidden(self, kuhn):
        maniac, rock = Spy("maniac", bots.maniac(kuhn)), Spy("rock", bots.rock(kuhn))
        play_match(kuhn, [maniac, rock], hands=6, rng=rng())
        # The rock folds to every bet, so no hand reaches a showdown.
        for view in maniac.views:
            theirs = [d for d in view.decisions if d.seat != view.seat]
            assert theirs and all(d.row is None for d in theirs)

    def test_a_showdown_reveals_the_players_in_it_to_everyone(self, kuhn3):
        maniac = Spy("maniac", bots.maniac(kuhn3))
        station = Spy("station", bots.calling_station(kuhn3))
        rock = Spy("rock", bots.rock(kuhn3))
        play_match(kuhn3, [maniac, station, rock], hands=1, rng=rng(), seatings=[(0, 1, 2)])
        # The maniac bets, the station calls, the rock folds: a showdown between seats 0 and 1.
        for spy in (maniac, station, rock):
            shown = {d.seat: d.row is not None for d in spy.views[0].decisions}
            assert shown[0] and shown[1]
        assert maniac.views[0].decisions[2].row is None      # the rock's fold, seen by others
        assert rock.views[0].decisions[2].row is not None    # the rock knows its own hand

    def test_reveal_all_turns_every_card_over(self, kuhn):
        maniac, rock = Spy("maniac", bots.maniac(kuhn)), Spy("rock", bots.rock(kuhn))
        play_match(kuhn, [maniac, rock], hands=6, rng=rng(), reveal="all")
        for view in maniac.views + rock.views:
            assert all(d.row is not None for d in view.decisions)

    def test_a_revealed_row_matches_the_action_taken_there(self, kuhn3):
        spies = [Spy(f"spy{i}", bots.uniform_random(kuhn3)) for i in range(3)]
        play_match(kuhn3, spies, hands=30, rng=rng(), reveal="all")
        for view in spies[0].views:
            for d in view.decisions:
                assert kuhn3.infoset_player[d.row] == d.seat
                assert kuhn3.legal[d.row, d.action]


class TestGuards:
    def test_needs_one_agent_per_seat(self, kuhn3):
        with pytest.raises(ValueError, match="seats 3 players"):
            play_match(kuhn3, [Entrant("rock", bots.rock(kuhn3))] * 2, 1, rng())

    def test_rejects_an_unknown_reveal_mode(self, kuhn):
        table = [Entrant("rock", bots.rock(kuhn))] * 2
        with pytest.raises(ValueError, match="reveal"):
            play_match(kuhn, table, 1, rng(), reveal="some")

    def test_rejects_a_seating_that_leaves_someone_out(self, kuhn):
        table = [Entrant("rock", bots.rock(kuhn))] * 2
        with pytest.raises(ValueError, match="seating"):
            play_match(kuhn, table, 1, rng(), seatings=[(0, 0)])

    def test_rejects_an_agent_whose_strategy_is_not_valid(self, kuhn):
        broken = Spy("broken", np.zeros((kuhn.num_infosets, kuhn.num_actions)))
        with pytest.raises(ValueError, match="sum to 1"):
            play_match(kuhn, [broken, Spy("rock", bots.rock(kuhn))], 1, rng())


class TestPlayMatches:
    def table(self, kuhn):
        return lambda _rng: [Spy("a", bots.uniform_random(kuhn)), Spy("b", bots.maniac(kuhn))]

    def test_stacks_one_match_per_row(self, kuhn):
        runs = play_matches(kuhn, self.table(kuhn), hands=6, matches=4)
        assert runs.shape == (4, 2, 6)

    def test_passes_options_on_to_each_match(self, kuhn):
        seen = []

        def table(_rng):
            spies = [Spy("a", bots.maniac(kuhn)), Spy("b", bots.rock(kuhn))]
            seen.append(spies)
            return spies

        play_matches(kuhn, table, hands=2, matches=2, reveal="all")
        assert all(d.row is not None for spies in seen for d in spies[0].views[0].decisions)

    def test_each_match_gets_its_own_random_stream(self, kuhn):
        draws = []
        play_matches(kuhn, lambda r: draws.append(r.random()) or self.table(kuhn)(r), 1, matches=3)
        assert len(set(draws)) == 3

    def test_the_same_seed_gives_the_same_matches(self, kuhn):
        first, second = [], []
        for record in (first, second):

            def table(r, record=record):
                record.append(r.random())
                return self.table(kuhn)(r)

            play_matches(kuhn, table, 1, 3, seed=5)
        assert first == second
