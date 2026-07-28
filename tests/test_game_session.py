"""Tests for GameSession — replay logs must be isolated per hand.

Regression: hands used to append to one shared state list, so every hand's
replay contained all previous hands concatenated.
"""

import random

import pytest

from pokerion.server.state import GameSession


def play_to_terminal(session: GameSession):
    """Drive a hand to completion. Human (player 0) always checks or calls."""
    while not session.is_terminal:
        state = session.get_state(viewer=session.human_player)
        if state["current_player"] != session.human_player:
            break  # safety net — agent should already have acted
        legal = state["legal_actions"]
        session.apply_action("check" if "check" in legal else "call")


@pytest.fixture
def session():
    # Empty strategy -> agent falls back to uniform random. Only structure matters here.
    random.seed(1234)
    return GameSession(variant="kuhn", strategy={})


class TestFreshSession:
    def test_starts_with_a_single_hand(self, session):
        assert len(session.hands) == 1

    def test_first_state_is_the_deal(self, session):
        replay = session.get_replay()
        assert len(replay) == 1
        assert replay[0]["action_history"] == []
        assert replay[0]["is_terminal"] is False


class TestPerHandIsolation:
    def test_replay_contains_exactly_one_terminal(self, session):
        play_to_terminal(session)
        replay = session.get_replay()
        assert sum(1 for s in replay if s["is_terminal"]) == 1
        assert replay[-1]["is_terminal"] is True

    def test_new_hand_replay_excludes_the_previous_hand(self, session):
        play_to_terminal(session)
        first = list(session.get_replay())

        session.new_hand()
        play_to_terminal(session)
        second = session.get_replay()

        # Under the old bug this was exactly 2, and `second` began with `first`.
        assert sum(1 for s in second if s["is_terminal"]) == 1
        assert second[: len(first)] != first

    def test_replay_length_matches_its_own_actions(self, session):
        for _ in range(3):
            play_to_terminal(session)
            replay = session.get_replay()
            # one state for the deal, plus one per action taken in this hand
            assert len(replay) == 1 + len(replay[-1]["action_history"])
            session.new_hand()

    def test_hands_do_not_share_a_state_list(self, session):
        play_to_terminal(session)
        session.new_hand()
        play_to_terminal(session)

        assert len(session.hands) == 2
        assert session.hands[0] is not session.hands[1]


class TestSessionHistory:
    def test_get_all_hands_returns_one_log_per_hand(self, session):
        for i in range(3):
            play_to_terminal(session)
            if i < 2:
                session.new_hand()

        hands = session.get_all_hands()
        assert len(hands) == 3
        for log in hands:
            assert log[-1]["is_terminal"] is True
            assert sum(1 for s in log if s["is_terminal"]) == 1

    def test_get_replay_returns_the_current_hand(self, session):
        play_to_terminal(session)
        session.new_hand()
        play_to_terminal(session)

        assert session.get_replay() is session.get_all_hands()[-1]


class TestReplayVisibility:
    def test_replay_is_god_mode(self, session):
        """Replay logs are recorded without a viewer — both cards always visible."""
        play_to_terminal(session)
        for state in session.get_replay():
            for player in state["players"]:
                assert player["card"] != "?"
