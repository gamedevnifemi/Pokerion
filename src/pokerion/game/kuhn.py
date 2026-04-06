"""Kuhn Poker — the simplest non-trivial poker game.

Rules:
- 3 cards: J (0), Q (1), K (2)
- 2 players, each antes 1 chip
- Each dealt 1 card
- Player 0 acts first: check or bet(1)
- Player 1 responds: check/bet or call/fold
- If player 0 checked and player 1 bet, player 0 can call or fold
- Showdown: higher card wins the pot

12 information sets, known analytical Nash equilibrium (game value = -1/18).
"""

import random
from itertools import permutations

from pokerion.common.types import Action, InfoSetKey, Player
from pokerion.game.base import History

CARDS = ["J", "Q", "K"]
CARD_RANK = {"J": 0, "Q": 1, "K": 2}


class KuhnHistory(History):
    """Immutable game tree node for Kuhn Poker.

    Internal encoding:
        cards: tuple of 2 cards dealt (cards[0] = player 0's card)
        actions: tuple of player actions taken so far ("check", "bet", "call", "fold")
    """

    def __init__(
        self,
        cards: tuple[str, ...] = (),
        actions: tuple[str, ...] = (),
    ):
        self._cards = cards
        self._actions = actions

    @property
    def variant(self) -> str:
        return "kuhn"

    @property
    def num_players(self) -> int:
        return 2

    def is_chance(self) -> bool:
        return len(self._cards) < 2

    def chance_actions(self) -> list[tuple[Action, float]]:
        if len(self._cards) == 0:
            # Deal both cards at once — all 6 permutations of 2 from 3
            deals = list(permutations(CARDS, 2))
            prob = 1.0 / len(deals)
            return [(f"{a}{b}", prob) for a, b in deals]
        # Should not be called if cards are already dealt
        return []

    def sample_chance(self) -> Action:
        if len(self._cards) == 0:
            a, b = random.sample(CARDS, 2)
            return f"{a}{b}"
        return ""

    def is_terminal(self) -> bool:
        if len(self._cards) < 2:
            return False
        a = self._actions
        # Both check -> showdown
        if a == ("check", "check"):
            return True
        # Bet then fold
        if len(a) >= 2 and a[-1] == "fold":
            return True
        # Bet then call
        if len(a) >= 2 and a[-2] == "bet" and a[-1] == "call":
            return True
        return False

    def terminal_utility(self, player: Player) -> float:
        a = self._actions
        p0_card = CARD_RANK[self._cards[0]]
        p1_card = CARD_RANK[self._cards[1]]
        winner = 0 if p0_card > p1_card else 1

        if a == ("check", "check"):
            # Pot = 2 (1 ante each). Winner gains 1.
            return 1.0 if player == winner else -1.0

        if a == ("bet", "fold"):
            # Player 0 bet, player 1 folded. Player 0 wins pot of 2.
            return 1.0 if player == 0 else -1.0

        if a == ("bet", "call"):
            # Pot = 4. Winner gains 2.
            return 2.0 if player == winner else -2.0

        if a == ("check", "bet", "fold"):
            # Player 0 checked, player 1 bet, player 0 folded. Player 1 wins.
            return 1.0 if player == 1 else -1.0

        if a == ("check", "bet", "call"):
            # Pot = 4. Showdown.
            return 2.0 if player == winner else -2.0

        raise ValueError(f"Not terminal: {a}")

    def active_player(self) -> Player:
        # After deal, player 0 acts on even action counts, player 1 on odd
        return len(self._actions) % 2

    def actions(self) -> list[Action]:
        if len(self._actions) == 0:
            # Player 0's first action
            return ["check", "bet"]
        last = self._actions[-1]
        if last == "check" and len(self._actions) == 1:
            # Player 1 after player 0 checked
            return ["check", "bet"]
        if last == "bet":
            # Facing a bet: call or fold
            return ["call", "fold"]
        return []

    def info_set_key(self) -> InfoSetKey:
        player = self.active_player()
        card = self._cards[player]
        action_str = ":".join(self._actions) if self._actions else ""
        return f"{card}|{action_str}" if action_str else card

    def __add__(self, action: Action) -> "KuhnHistory":
        if self.is_chance():
            # Action is a deal string like "KQ"
            return KuhnHistory(cards=(action[0], action[1]), actions=())
        return KuhnHistory(cards=self._cards, actions=self._actions + (action,))

    def to_state_dict(self, viewer: Player | None = None) -> dict:
        players = []
        for i in range(2):
            card = None
            if len(self._cards) > i:
                if viewer is None or viewer == i or self.is_terminal():
                    card = self._cards[i]
                else:
                    card = "?"
            players.append({
                "id": i,
                "card": card,
                "stack": 1,
            })

        # Compute pot from actions
        pot = 2  # antes
        bets = [0, 0]
        for idx, a in enumerate(self._actions):
            acting = idx % 2
            if a == "bet":
                bets[acting] = 1
            elif a == "call":
                bets[acting] = 1
        pot += sum(bets)

        action_history = []
        for idx, a in enumerate(self._actions):
            action_history.append({"player": idx % 2, "action": a})

        winner = None
        if self.is_terminal():
            # Determine winner from utility
            u0 = self.terminal_utility(0)
            if u0 > 0:
                winner = 0
            else:
                winner = 1

        return {
            "variant": "kuhn",
            "players": players,
            "pot": pot,
            "community_cards": [],
            "action_history": action_history,
            "legal_actions": self.actions() if not self.is_terminal() and not self.is_chance() else [],
            "current_player": self.active_player() if not self.is_terminal() and not self.is_chance() else None,
            "is_terminal": self.is_terminal(),
            "is_chance": self.is_chance(),
            "winner": winner,
            "round": 0,
        }

    def __repr__(self) -> str:
        cards = "".join(self._cards) if self._cards else "??"
        actions = ",".join(self._actions) if self._actions else "-"
        return f"Kuhn({cards}|{actions})"
