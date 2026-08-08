"""Tests for the Match model — chips, seat rotation, bounded length, validation.

Replaces test_game_session.py: GameSession became Match when the game gained
chips, alternating seats, and a fixed length with a winner decided by chips.
"""

import random

import pytest

from pokerion.server.state import Match, MatchError


def play_hand(match: Match) -> dict:
    """Drive the current hand to completion. Human always checks or calls."""
    record = None
    while record is None:
        legal = match.current.actions()
        record = match.apply_action("check" if "check" in legal else "call")
    return record


@pytest.fixture
def match():
    # Empty strategy -> agent plays uniform random. Structure is what's tested.
    random.seed(1234)
    return Match(variant="kuhn", strategy={}, length=4)


class TestLength:
    def test_odd_length_is_forced_even(self):
        m = Match(variant="kuhn", strategy={}, length=5)
        assert m.length == 6  # odd gifts one side the losing seat an extra time

    def test_length_is_clamped(self):
        assert Match(variant="kuhn", strategy={}, length=0).length == Match.MIN_LENGTH
        assert Match(variant="kuhn", strategy={}, length=10_000).length == Match.MAX_LENGTH

    def test_default_length(self):
        assert Match(variant="kuhn", strategy={}).length == Match.DEFAULT_LENGTH


class TestSeatRotation:
    def test_human_opens_in_seat_zero(self, match):
        assert match.human_seat == 0

    def test_seat_alternates_each_hand(self, match):
        seats = []
        for _ in range(match.length):
            seats.append(match.human_seat)
            play_hand(match)
            if not match.is_complete:
                match.new_hand()
        assert seats == [0, 1, 0, 1]

    def test_seat_holds_through_terminal_until_next_deal(self, match):
        play_hand(match)
        # Hand settled (hand_index moved on) but the table must not re-orient
        # until the next deal.
        assert match.human_seat == 0
        match.new_hand()
        assert match.human_seat == 1

    def test_agent_opens_when_human_in_seat_one(self, match):
        play_hand(match)
        match.new_hand()
        # Human is seat 1: the agent (seat 0) must have already acted or the
        # human must be next — never "waiting on the agent".
        assert (
            match.current.is_terminal()
            or match.current.active_player() == match.human_seat
        )


class TestChips:
    def test_chips_equal_sum_of_hand_deltas(self, match):
        while not match.is_complete:
            play_hand(match)
            if not match.is_complete:
                match.new_hand()
        assert match.human_chips == pytest.approx(
            sum(h["chips_delta"] for h in match.hands)
        )

    def test_delta_matches_terminal_utility_for_human_seat(self, match):
        record = play_hand(match)
        assert record["chips_delta"] == match.current.terminal_utility(
            record["human_seat"]
        )

    def test_winner_is_decided_by_chips(self, match):
        while not match.is_complete:
            play_hand(match)
            if not match.is_complete:
                match.new_hand()
        expected = (
            "human" if match.human_chips > 0
            else "agent" if match.human_chips < 0
            else "draw"
        )
        assert match.winner == expected

    def test_no_winner_before_completion(self, match):
        assert match.winner is None


class TestHandRecords:
    def test_one_terminal_state_per_hand(self, match):
        record = play_hand(match)
        terminals = [s for s in record["states"] if s["is_terminal"]]
        assert len(terminals) == 1
        assert record["states"][-1]["is_terminal"]

    def test_records_carry_seat_and_winner(self, match):
        record = play_hand(match)
        assert record["human_seat"] in (0, 1)
        assert record["winner_seat"] in (0, 1)

    def test_hands_do_not_share_state_lists(self, match):
        play_hand(match)
        match.new_hand()
        play_hand(match)
        assert match.hands[0]["states"] is not match.hands[1]["states"]


class TestValidation:
    def test_illegal_action_rejected(self, match):
        with pytest.raises(MatchError):
            match.apply_action("fold")  # nothing to fold to at the open

    def test_action_after_terminal_rejected(self, match):
        play_hand(match)
        with pytest.raises(MatchError):
            match.apply_action("check")

    def test_new_hand_mid_hand_rejected(self, match):
        with pytest.raises(MatchError):
            match.new_hand()

    def test_play_after_match_complete_rejected(self, match):
        while not match.is_complete:
            play_hand(match)
            if not match.is_complete:
                match.new_hand()
        with pytest.raises(MatchError):
            match.new_hand()


class TestStateView:
    def test_state_hides_agent_card_and_shows_match(self, match):
        state = match.get_state()
        assert state["players"][match.agent_seat]["card"] == "?"
        assert state["players"][match.human_seat]["card"] in ("J", "Q", "K")
        assert state["match"]["length"] == 4
        assert state["viewer"] == match.human_seat

    def test_committed_tracks_ante(self, match):
        state = match.get_state()
        assert all(p["committed"] >= 1 for p in state["players"])
