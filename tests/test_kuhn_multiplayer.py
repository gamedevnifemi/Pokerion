"""Tests for Kuhn poker beyond the standard two-player game.

The three-player rules and the worked example come from Szafron, Gibson and
Sturtevant, "A Parameterized Family of Equilibrium Profiles for Three-Player
Kuhn Poker" (AAMAS 2013), section 2. That paper numbers the cards 1 to 4; here
they are J, Q, K, A.
"""

from collections import Counter
from itertools import permutations, product

import pytest

from pokerion.game.kuhn import KUHN_2P, KUHN_3P, KuhnHistory, KuhnRules


def play(rules: KuhnRules, deal: str, *actions: str) -> KuhnHistory:
    history = rules.root() + deal
    for action in actions:
        history = history + action
    return history


def information_sets(rules: KuhnRules) -> dict[str, int]:
    """Every information set key in the game, with the seat that acts there."""
    found: dict[str, int] = {}

    def walk(history: KuhnHistory) -> None:
        if history.is_terminal():
            return
        if history.is_chance():
            for action, _ in history.chance_actions():
                walk(history + action)
            return
        found[history.info_set_key()] = history.active_player()
        for action in history.actions():
            walk(history + action)

    walk(rules.root())
    return found


def terminal_histories(rules: KuhnRules) -> list[KuhnHistory]:
    found: list[KuhnHistory] = []

    def walk(history: KuhnHistory) -> None:
        if history.is_terminal():
            found.append(history)
        elif history.is_chance():
            for action, _ in history.chance_actions():
                walk(history + action)
        else:
            for action in history.actions():
                walk(history + action)

    walk(rules.root())
    return found


class TestRules:
    def test_default_history_is_the_standard_two_player_game(self):
        assert KuhnHistory().rules == KUHN_2P
        assert KuhnHistory().variant == "kuhn"
        assert KuhnHistory().num_players == 2

    def test_other_games_get_a_descriptive_variant(self):
        assert KUHN_3P.root().variant == "kuhn-3p-4c"
        assert KUHN_3P.root().num_players == 3

    def test_needs_more_cards_than_players(self):
        with pytest.raises(ValueError, match="at least 4 cards"):
            KuhnRules(num_players=3, deck=("J", "Q", "K"))

    def test_needs_two_players(self):
        with pytest.raises(ValueError, match="at least two players"):
            KuhnRules(num_players=1)

    def test_rejects_duplicate_cards(self):
        with pytest.raises(ValueError, match="unique"):
            KuhnRules(deck=("J", "J", "K"))

    @pytest.mark.parametrize("bad", ["", "a,b", "a|b", "a:b"])
    def test_rejects_card_names_that_clash_with_separators(self, bad):
        with pytest.raises(ValueError, match="card name"):
            KuhnRules(deck=("J", "Q", bad))

    def test_rank_follows_deck_order(self):
        assert [KUHN_3P.rank(card) for card in "JQKA"] == [0, 1, 2, 3]


class TestThreePlayerDeal:
    def test_twenty_four_equally_likely_deals(self):
        deals = KUHN_3P.root().chance_actions()
        assert len(deals) == 24  # ordered choices of 3 cards from 4
        assert all(prob == pytest.approx(1 / 24) for _, prob in deals)
        assert len({action for action, _ in deals}) == 24

    def test_deal_gives_each_seat_its_card(self):
        state = (KUHN_3P.root() + "JQA").to_state_dict()
        assert [p["card"] for p in state["players"]] == ["J", "Q", "A"]

    def test_sampled_deal_is_a_legal_deal(self):
        root = KUHN_3P.root()
        legal = {action for action, _ in root.chance_actions()}
        assert all(root.sample_chance() in legal for _ in range(50))


class TestThreePlayerBetting:
    def test_seats_act_in_order(self):
        assert play(KUHN_3P, "JQA").active_player() == 0
        assert play(KUHN_3P, "JQA", "check").active_player() == 1
        assert play(KUHN_3P, "JQA", "check", "check").active_player() == 2

    def test_action_wraps_round_to_players_who_checked(self):
        # Seat 2 bets after two checks, so seats 0 and 1 still have to answer.
        history = play(KUHN_3P, "JQA", "check", "check", "bet")
        assert history.active_player() == 0
        assert history.actions() == ["call", "fold"]
        assert (history + "fold").active_player() == 1

    def test_check_or_bet_until_someone_bets(self):
        assert play(KUHN_3P, "JQA", "check").actions() == ["check", "bet"]
        assert play(KUHN_3P, "JQA", "bet").actions() == ["call", "fold"]
        assert play(KUHN_3P, "JQA", "bet", "fold").actions() == ["call", "fold"]

    def test_three_checks_end_the_hand(self):
        assert not play(KUHN_3P, "JQA", "check", "check").is_terminal()
        assert play(KUHN_3P, "JQA", "check", "check", "check").is_terminal()

    def test_hand_ends_once_everyone_has_answered_the_bet(self):
        for first, second in product(["call", "fold"], repeat=2):
            for prefix in [(), ("check",), ("check", "check")]:
                bet = play(KUHN_3P, "JQA", *prefix, "bet")
                assert not (bet + first).is_terminal()
                assert (bet + first + second).is_terminal()

    def test_thirteen_ways_a_hand_can_end(self):
        # All check, or one of three seats bets and the other two each call or fold.
        endings = {h._actions for h in terminal_histories(KUHN_3P)}
        assert len(endings) == 1 + 3 * 4

    def test_no_actions_once_the_hand_is_over(self):
        assert play(KUHN_3P, "JQA", "bet", "fold", "fold").actions() == []


class TestThreePlayerPayoffs:
    def payoffs(self, deal: str, *actions: str) -> list[float]:
        history = play(KUHN_3P, deal, *actions)
        return [history.terminal_utility(seat) for seat in range(3)]

    def test_worked_example_from_the_paper(self):
        # Cards 1, 2, 4. P1 checks, P2 bets, P3 folds, P1 calls: P2 wins a pot of 5.
        assert self.payoffs("JQA", "check", "bet", "fold", "call") == [-2.0, 3.0, -1.0]

    def test_all_check_highest_card_takes_the_antes(self):
        assert self.payoffs("JQA", "check", "check", "check") == [-1.0, -1.0, 2.0]

    def test_bet_and_everyone_folds(self):
        assert self.payoffs("JQA", "bet", "fold", "fold") == [2.0, -1.0, -1.0]

    def test_folded_hand_cannot_win(self):
        # The ace folds, so the queen beats the jack at showdown.
        assert self.payoffs("JQA", "bet", "call", "fold") == [-2.0, 3.0, -1.0]

    def test_three_way_showdown(self):
        assert self.payoffs("JQA", "bet", "call", "call") == [-2.0, -2.0, 4.0]

    def test_every_ending_is_zero_sum(self):
        for history in terminal_histories(KUHN_3P):
            assert sum(history.terminal_utility(seat) for seat in range(3)) == 0.0

    def test_payoff_before_the_end_is_an_error(self):
        with pytest.raises(ValueError, match="Not terminal"):
            play(KUHN_3P, "JQA", "bet").terminal_utility(0)


class TestThreePlayerInformationSets:
    def test_forty_eight_information_sets_sixteen_per_seat(self):
        # Four cards times four betting situations for each seat (Table 1 of the paper).
        found = information_sets(KUHN_3P)
        assert len(found) == 48
        assert Counter(found.values()) == {0: 16, 1: 16, 2: 16}

    def test_a_player_cannot_see_the_other_cards(self):
        assert play(KUHN_3P, "JQA", "check").info_set_key() == "Q|check"
        assert play(KUHN_3P, "KQJ", "check").info_set_key() == "Q|check"

    def test_key_records_the_whole_betting_so_far(self):
        history = play(KUHN_3P, "JQA", "check", "check", "bet")
        assert history.info_set_key() == "J|check:check:bet"


class TestThreePlayerStateDict:
    def test_lists_every_seat_and_who_did_what(self):
        state = play(KUHN_3P, "JQA", "check", "bet", "fold").to_state_dict()
        assert [p["id"] for p in state["players"]] == [0, 1, 2]
        assert state["action_history"] == [
            {"player": 0, "action": "check"},
            {"player": 1, "action": "bet"},
            {"player": 2, "action": "fold"},
        ]
        assert state["current_player"] == 0
        assert state["pot"] == 4
        assert [p["committed"] for p in state["players"]] == [1, 2, 1]

    def test_viewer_sees_only_their_own_card_during_the_hand(self):
        state = play(KUHN_3P, "JQA", "check").to_state_dict(viewer=1)
        assert [p["card"] for p in state["players"]] == ["?", "Q", "?"]

    def test_winner_is_a_seat_number(self):
        state = play(KUHN_3P, "JQA", "check", "bet", "fold", "call").to_state_dict()
        assert state["is_terminal"] is True
        assert state["winner"] == 1


class TestOtherDecks:
    def test_long_card_names_survive_the_deal(self):
        rules = KuhnRules(deck=tuple(str(n) for n in range(2, 15)))
        deals = [action for action, _ in rules.root().chance_actions()]
        assert "10,14" in deals
        assert (rules.root() + "10,14").to_state_dict()["players"][1]["card"] == "14"

    def test_information_sets_grow_with_the_deck(self):
        # Four betting situations per card, for two players.
        assert len(information_sets(KuhnRules(deck=tuple("23456789")))) == 4 * 8

    def test_two_player_deal_strings_are_unchanged(self):
        deals = [action for action, _ in KUHN_2P.root().chance_actions()]
        assert sorted(deals) == sorted("".join(cards) for cards in permutations("JQK", 2))
