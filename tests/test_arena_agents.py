"""Tests for the adapting agent and its two swappable parts."""

import numpy as np
import pytest

from pokerion.arena import (
    AdaptiveAgent,
    CountingMemory,
    Decision,
    Entrant,
    HandView,
    MixWithBestResponse,
    PlayTheCore,
    best_reply_to_table,
    bots,
    evaluate_table,
    play_match,
    play_matches,
)
from pokerion.evaluation import best_response


def hand(kuhn, seating, seat, *decisions):
    """A hand view built by hand: decisions are (seat, information set key or None, action name)."""
    return HandView(
        seating=seating,
        seat=seat,
        decisions=tuple(
            Decision(seat=s, action=kuhn.column(a), row=None if key is None else kuhn.row(key))
            for s, key, a in decisions
        ),
    )


def probability(tree, strategy, key, action):
    return strategy[tree.row(key), tree.column(action)]


class TestCountingMemory:
    def memory(self, kuhn, weight=5.0):
        memory = CountingMemory(kuhn.uniform_strategy(), prior_weight=weight)
        memory.begin_match(kuhn, me=0)
        return memory

    def test_predicts_the_prior_before_seeing_anything(self, kuhn):
        assert np.array_equal(self.memory(kuhn).predict(1), kuhn.uniform_strategy())

    def test_moves_from_the_prior_towards_what_it_sees(self, kuhn):
        memory = self.memory(kuhn)
        for _ in range(5):
            memory.observe(hand(kuhn, (0, 1), 0, (0, "K", "bet"), (1, "Q|bet", "fold")))
        # Five imaginary visits at one half, five real folds: (5 * 0.5 + 5) / (5 + 5).
        assert probability(kuhn, memory.predict(1), "Q|bet", "fold") == pytest.approx(0.75)
        assert probability(kuhn, memory.predict(1), "Q|bet", "call") == pytest.approx(0.25)

    def test_other_information_sets_stay_at_the_prior(self, kuhn):
        memory = self.memory(kuhn)
        memory.observe(hand(kuhn, (0, 1), 0, (0, "K", "bet"), (1, "Q|bet", "fold")))
        assert probability(kuhn, memory.predict(1), "J|bet", "fold") == 0.5

    def test_a_heavier_prior_moves_more_slowly(self, kuhn):
        light, heavy = self.memory(kuhn, 1.0), self.memory(kuhn, 50.0)
        for memory in (light, heavy):
            memory.observe(hand(kuhn, (0, 1), 0, (0, "K", "bet"), (1, "Q|bet", "fold")))
        assert probability(kuhn, light.predict(1), "Q|bet", "fold") == pytest.approx(0.75)
        assert probability(kuhn, heavy.predict(1), "Q|bet", "fold") < 0.52

    def test_ignores_its_own_actions(self, kuhn):
        memory = self.memory(kuhn)
        memory.observe(hand(kuhn, (0, 1), 0, (0, "K", "bet"), (1, "Q|bet", "fold")))
        assert np.array_equal(memory.predict(0), kuhn.uniform_strategy())

    def test_cannot_count_an_action_whose_cards_stayed_hidden(self, kuhn):
        memory = self.memory(kuhn)
        memory.observe(hand(kuhn, (0, 1), 0, (0, "K", "bet"), (1, None, "fold")))
        assert np.array_equal(memory.predict(1), kuhn.uniform_strategy())

    def test_follows_an_opponent_from_seat_to_seat(self, kuhn):
        memory = self.memory(kuhn)
        memory.observe(hand(kuhn, (0, 1), 0, (0, "K", "bet"), (1, "Q|bet", "fold")))
        memory.observe(hand(kuhn, (1, 0), 1, (0, "Q", "bet"), (1, "K|bet", "call")))
        prediction = memory.predict(1)
        assert probability(kuhn, prediction, "Q|bet", "fold") > 0.5  # seen when it sat second
        assert probability(kuhn, prediction, "Q", "bet") > 0.5       # seen when it sat first

    def test_keeps_opponents_apart(self, kuhn3):
        memory = CountingMemory(kuhn3.uniform_strategy())
        memory.begin_match(kuhn3, me=0)
        view = HandView(
            seating=(0, 1, 2),
            seat=0,
            decisions=(
                Decision(seat=0, action=kuhn3.column("check"), row=kuhn3.row("J")),
                Decision(seat=1, action=kuhn3.column("bet"), row=kuhn3.row("Q|check")),
                Decision(seat=2, action=kuhn3.column("fold"), row=kuhn3.row("A|check:bet")),
            ),
        )
        memory.observe(view)
        assert probability(kuhn3, memory.predict(1), "Q|check", "bet") > 0.5
        assert probability(kuhn3, memory.predict(2), "Q|check", "bet") == 0.5

    def test_forgets_between_matches(self, kuhn):
        memory = self.memory(kuhn)
        memory.observe(hand(kuhn, (0, 1), 0, (0, "K", "bet"), (1, "Q|bet", "fold")))
        memory.begin_match(kuhn, me=0)
        assert np.array_equal(memory.predict(1), kuhn.uniform_strategy())

    def test_needs_a_positive_prior_weight(self, kuhn):
        with pytest.raises(ValueError, match="prior_weight"):
            CountingMemory(kuhn.uniform_strategy(), prior_weight=0.0)


class TestExploitationRules:
    def earnings(self, kuhn, chosen, table, seat):
        """What `seat` wins playing its rows of `chosen` against the rest of `table`."""
        profile = table.copy()
        rows = kuhn.infoset_player == seat
        profile[rows] = chosen[rows]
        return kuhn.expected_utilities(profile)[seat]

    def test_play_the_core_ignores_the_table(self, kuhn, kuhn_equilibrium):
        chosen = PlayTheCore().choose(kuhn, kuhn_equilibrium, bots.rock(kuhn), seat=0)
        assert chosen is kuhn_equilibrium

    @pytest.mark.parametrize("seat", [0, 1])
    def test_full_weight_earns_the_best_response_value(self, kuhn, kuhn_equilibrium, seat):
        table = bots.rock(kuhn)
        chosen = MixWithBestResponse(1.0).choose(kuhn, kuhn_equilibrium, table, seat)
        assert self.earnings(kuhn, chosen, table, seat) == pytest.approx(
            best_response(kuhn, table, seat).value
        )

    def test_zero_weight_is_the_core(self, kuhn, kuhn_equilibrium):
        chosen = MixWithBestResponse(0.0).choose(kuhn, kuhn_equilibrium, bots.rock(kuhn), seat=0)
        assert chosen is kuhn_equilibrium

    def test_earnings_rise_in_proportion_to_the_weight(self, kuhn, kuhn_equilibrium):
        table = bots.maniac(kuhn)
        core = self.earnings(kuhn, kuhn_equilibrium, table, 0)
        best = best_response(kuhn, table, 0).value
        chosen = MixWithBestResponse(0.25).choose(kuhn, kuhn_equilibrium, table, 0)
        assert self.earnings(kuhn, chosen, table, 0) == pytest.approx(0.75 * core + 0.25 * best)

    def test_rejects_a_weight_outside_zero_to_one(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            MixWithBestResponse(1.5)

    def test_tie_break_decides_what_it_tries_against_an_equilibrium_prediction(
        self, kuhn, kuhn_equilibrium
    ):
        # Predicting equilibrium play leaves the first player indifferent with a king.
        keen = MixWithBestResponse(1.0, prefer=("bet", "call"), tolerance=1e-3)
        shy = MixWithBestResponse(1.0, prefer=("check", "fold"), tolerance=1e-3)
        bet_king = (kuhn.row("K"), kuhn.column("bet"))
        assert keen.choose(kuhn, kuhn_equilibrium, kuhn_equilibrium, 0)[bet_king] == 1.0
        assert shy.choose(kuhn, kuhn_equilibrium, kuhn_equilibrium, 0)[bet_king] == 0.0


class TestAdaptiveAgent:
    """Two-player Kuhn, with every card shown after each hand unless a test says otherwise."""

    def agent(self, core, weight=1.0):
        rule = MixWithBestResponse(weight) if weight else PlayTheCore()
        return AdaptiveAgent("hero", core, CountingMemory(core), rule)

    def course(self, kuhn, core, opponent, hands, weight=1.0, matches=6, reveal="all"):
        """The hero's expected chips on each hand, averaged over several matches."""
        runs = play_matches(
            kuhn,
            lambda _rng: [self.agent(core, weight), opponent],
            hands,
            matches,
            seed=1,
            reveal=reveal,
        )
        return runs[:, 0, :].mean(axis=0)

    def test_must_be_in_a_match_before_it_plays(self, kuhn, kuhn_equilibrium):
        with pytest.raises(RuntimeError, match="begin_match"):
            self.agent(kuhn_equilibrium).strategy_for((0, 1))

    def test_with_the_core_rule_it_plays_exactly_the_core(self, kuhn, kuhn_equilibrium):
        rock = Entrant("rock", bots.rock(kuhn))
        hero, core = self.agent(kuhn_equilibrium, weight=0.0), Entrant("core", kuhn_equilibrium)
        adapting = play_match(kuhn, [hero, rock], 6, np.random.default_rng(0), reveal="all")
        fixed = play_match(kuhn, [core, rock], 6, np.random.default_rng(0), reveal="all")
        assert np.array_equal(adapting.per_hand, fixed.per_hand)

    def test_learns_to_take_a_rocks_ante_every_hand(self, kuhn, kuhn_equilibrium):
        course = self.course(kuhn, kuhn_equilibrium, Entrant("rock", bots.rock(kuhn)), hands=200)
        assert course[-50:].mean() > 0.9  # the ceiling is 1.0

    @pytest.mark.parametrize("make", [bots.rock, bots.calling_station, bots.maniac])
    def test_does_no_harm_against_a_weak_bot(self, kuhn, kuhn_equilibrium, make):
        opponent = Entrant("weak", make(kuhn))
        core_wins = evaluate_table(kuhn, [Entrant("core", kuhn_equilibrium), opponent]).winnings[0]
        course = self.course(kuhn, kuhn_equilibrium, opponent, hands=100)
        for checkpoint in (25, 100):
            assert course[:checkpoint].mean() >= core_wins

    def test_never_beats_the_ceiling(self, kuhn, kuhn_equilibrium):
        opponent = Entrant("maniac", bots.maniac(kuhn))
        ceiling = best_reply_to_table(kuhn, [Entrant("core", kuhn_equilibrium), opponent], hero=0)
        course = self.course(kuhn, kuhn_equilibrium, opponent, hands=100)
        # A pair of hands covers both seats, which is what the ceiling averages over.
        pairs = course.reshape(-1, 2).mean(axis=1)
        assert (pairs <= ceiling + 1e-9).all()

    def test_loses_almost_nothing_to_solid_play(self, kuhn, kuhn_equilibrium):
        course = self.course(kuhn, kuhn_equilibrium, Entrant("solid", kuhn_equilibrium), hands=100)
        assert course.mean() > -0.01

    def test_half_weight_takes_half_the_gain(self, kuhn, kuhn_equilibrium):
        opponent = Entrant("rock", bots.rock(kuhn))
        core_wins = evaluate_table(kuhn, [Entrant("core", kuhn_equilibrium), opponent]).winnings[0]
        runs = {
            w: play_matches(
                kuhn,
                lambda _rng, w=w: [self.agent(kuhn_equilibrium, w), opponent],
                hands=40,
                matches=1,
                seed=3,
                reveal="all",
                seatings=[(0, 1)],
            )[0, 0]
            for w in (0.5, 1.0)
        }
        # On the first hand both believe the same thing, so the half-weight agent
        # gains exactly half of what the full-weight agent gains over the core.
        both_core = kuhn.combine([kuhn_equilibrium, opponent.strategy])
        core_first_hand = kuhn.expected_utilities(both_core)[0]
        half_gain, full_gain = (runs[w][0] - core_first_hand for w in (0.5, 1.0))
        assert half_gain == pytest.approx(0.5 * full_gain)
        assert core_wins < runs[0.5].mean() < runs[1.0].mean()

    def test_is_blind_to_folds_when_only_showdowns_are_shown(self, kuhn, kuhn_equilibrium):
        # A known limit of counting: the rock's folds are never shown, so they are never learned.
        rock = Entrant("rock", bots.rock(kuhn))
        seeing_all = self.course(kuhn, kuhn_equilibrium, rock, hands=200)
        seeing_showdowns = self.course(kuhn, kuhn_equilibrium, rock, hands=200, reveal="showdown")
        assert seeing_all[-50:].mean() > 0.9
        assert seeing_showdowns[-50:].mean() < 0.5

    def test_adapts_to_two_opponents_at_once(self, kuhn3, kuhn3_equilibrium):
        rock, maniac = Entrant("rock", bots.rock(kuhn3)), Entrant("maniac", bots.maniac(kuhn3))
        core = Entrant("core", kuhn3_equilibrium)
        core_wins = evaluate_table(kuhn3, [core, rock, maniac]).winnings[0]
        runs = play_matches(
            kuhn3,
            lambda _rng: [self.agent(kuhn3_equilibrium), rock, maniac],
            hands=150,
            matches=3,
            seed=2,
            reveal="all",
        )
        assert np.abs(runs.sum(axis=1)).max() < 1e-9          # zero-sum on every hand
        assert runs[:, 0, -60:].mean() > core_wins + 0.3      # far ahead of not adapting
