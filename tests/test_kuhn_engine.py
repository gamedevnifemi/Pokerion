"""Tests for the Kuhn Poker game engine."""

import pytest

from pokerion.game.kuhn import CARDS, KuhnHistory


class TestChanceNode:
    def test_root_is_chance(self):
        h = KuhnHistory()
        assert h.is_chance()
        assert not h.is_terminal()

    def test_chance_actions_are_all_deals(self):
        h = KuhnHistory()
        deals = h.chance_actions()
        assert len(deals) == 6  # P(3,2) = 6
        # All probabilities equal
        for _, prob in deals:
            assert prob == pytest.approx(1 / 6)
        # All unique
        actions = [a for a, _ in deals]
        assert len(set(actions)) == 6

    def test_after_deal_not_chance(self):
        h = KuhnHistory() + "KQ"
        assert not h.is_chance()
        assert not h.is_terminal()


class TestActions:
    def test_player0_opens(self):
        h = KuhnHistory() + "KQ"
        assert h.active_player() == 0
        assert h.actions() == ["check", "bet"]

    def test_player1_after_check(self):
        h = KuhnHistory() + "KQ" + "check"
        assert h.active_player() == 1
        assert h.actions() == ["check", "bet"]

    def test_player1_after_bet(self):
        h = KuhnHistory() + "KQ" + "bet"
        assert h.active_player() == 1
        assert h.actions() == ["call", "fold"]

    def test_player0_after_check_bet(self):
        h = KuhnHistory() + "KQ" + "check" + "bet"
        assert h.active_player() == 0
        assert h.actions() == ["call", "fold"]


class TestTerminal:
    def test_check_check_is_terminal(self):
        h = KuhnHistory() + "KQ" + "check" + "check"
        assert h.is_terminal()

    def test_bet_fold_is_terminal(self):
        h = KuhnHistory() + "KQ" + "bet" + "fold"
        assert h.is_terminal()

    def test_bet_call_is_terminal(self):
        h = KuhnHistory() + "KQ" + "bet" + "call"
        assert h.is_terminal()

    def test_check_bet_fold_is_terminal(self):
        h = KuhnHistory() + "KQ" + "check" + "bet" + "fold"
        assert h.is_terminal()

    def test_check_bet_call_is_terminal(self):
        h = KuhnHistory() + "KQ" + "check" + "bet" + "call"
        assert h.is_terminal()

    def test_single_check_not_terminal(self):
        h = KuhnHistory() + "KQ" + "check"
        assert not h.is_terminal()

    def test_single_bet_not_terminal(self):
        h = KuhnHistory() + "KQ" + "bet"
        assert not h.is_terminal()


class TestUtilities:
    """Utilities are from the perspective of the requested player.
    Pot starts at 2 (1 ante each). Bet adds 1."""

    def test_check_check_higher_wins(self):
        # K vs Q, both check. K wins. Pot=2, gain=1.
        h = KuhnHistory() + "KQ" + "check" + "check"
        assert h.terminal_utility(0) == 1.0   # K wins
        assert h.terminal_utility(1) == -1.0

    def test_check_check_lower_loses(self):
        # J vs K, both check. K wins.
        h = KuhnHistory() + "JK" + "check" + "check"
        assert h.terminal_utility(0) == -1.0  # J loses
        assert h.terminal_utility(1) == 1.0

    def test_bet_fold_bettor_wins(self):
        # Player 0 bets, player 1 folds. Player 0 wins pot of 2.
        h = KuhnHistory() + "JK" + "bet" + "fold"
        assert h.terminal_utility(0) == 1.0   # wins regardless of cards
        assert h.terminal_utility(1) == -1.0

    def test_bet_call_showdown(self):
        # K vs J, bet-call. Pot=4, K wins 2.
        h = KuhnHistory() + "KJ" + "bet" + "call"
        assert h.terminal_utility(0) == 2.0
        assert h.terminal_utility(1) == -2.0

    def test_check_bet_fold(self):
        # Player 0 checks, player 1 bets, player 0 folds. Player 1 wins.
        h = KuhnHistory() + "QK" + "check" + "bet" + "fold"
        assert h.terminal_utility(0) == -1.0
        assert h.terminal_utility(1) == 1.0

    def test_check_bet_call_showdown(self):
        # Q vs J, check-bet-call. Pot=4, Q wins 2.
        h = KuhnHistory() + "QJ" + "check" + "bet" + "call"
        assert h.terminal_utility(0) == 2.0
        assert h.terminal_utility(1) == -2.0

    def test_utilities_are_zero_sum(self):
        """For every terminal state, u(p0) + u(p1) = 0."""
        from itertools import permutations

        action_sequences = [
            ("check", "check"),
            ("bet", "fold"),
            ("bet", "call"),
            ("check", "bet", "fold"),
            ("check", "bet", "call"),
        ]
        for cards in permutations(CARDS, 2):
            deal = "".join(cards)
            for actions in action_sequences:
                h = KuhnHistory()
                h = h + deal
                for a in actions:
                    h = h + a
                assert h.is_terminal()
                assert h.terminal_utility(0) + h.terminal_utility(1) == 0.0


class TestInfoSets:
    def test_hides_opponent_card(self):
        h1 = KuhnHistory() + "KQ"
        h2 = KuhnHistory() + "KJ"
        # Player 0 has K in both — same info set before any actions
        assert h1.info_set_key() == h2.info_set_key()

    def test_different_cards_different_info_set(self):
        h1 = KuhnHistory() + "KQ"
        h2 = KuhnHistory() + "QK"
        assert h1.info_set_key() != h2.info_set_key()

    def test_actions_in_info_set(self):
        h = KuhnHistory() + "KQ" + "check"
        # Player 1 sees own card (Q) and the check
        assert h.info_set_key() == "Q|check"

    def test_12_unique_info_sets(self):
        """Kuhn Poker has exactly 12 information sets."""
        info_sets = set()
        from itertools import permutations

        def traverse(h: KuhnHistory):
            if h.is_terminal():
                return
            if h.is_chance():
                for action, _ in h.chance_actions():
                    traverse(h + action)
            else:
                info_sets.add(h.info_set_key())
                for action in h.actions():
                    traverse(h + action)

        traverse(KuhnHistory())
        assert len(info_sets) == 12


class TestStateDict:
    def test_initial_state(self):
        h = KuhnHistory() + "KQ"
        state = h.to_state_dict()
        assert state["variant"] == "kuhn"
        assert state["is_chance"] is False
        assert state["is_terminal"] is False
        assert state["current_player"] == 0
        assert state["legal_actions"] == ["check", "bet"]
        assert state["pot"] == 2  # antes only

    def test_hides_opponent_card(self):
        h = KuhnHistory() + "KQ"
        state = h.to_state_dict(viewer=0)
        assert state["players"][0]["card"] == "K"
        assert state["players"][1]["card"] == "?"  # hidden

    def test_shows_all_in_god_mode(self):
        h = KuhnHistory() + "KQ"
        state = h.to_state_dict(viewer=None)
        assert state["players"][0]["card"] == "K"
        assert state["players"][1]["card"] == "Q"

    def test_terminal_shows_all(self):
        h = KuhnHistory() + "KQ" + "check" + "check"
        state = h.to_state_dict(viewer=0)
        # At terminal, both cards revealed
        assert state["players"][1]["card"] == "Q"
        assert state["is_terminal"] is True
        assert state["winner"] == 0  # K > Q


class TestImmutability:
    def test_add_does_not_mutate(self):
        h1 = KuhnHistory() + "KQ"
        h2 = h1 + "check"
        # h1 should still be at the deal node
        assert h1.active_player() == 0
        assert h1.actions() == ["check", "bet"]
        # h2 should be after check
        assert h2.active_player() == 1
