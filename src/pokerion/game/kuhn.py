"""Kuhn poker — the simplest non-trivial poker game, for two or more players.

Rules, for any number of players:
- Every player antes 1 chip and is dealt one card from a small deck.
- Players act in seat order. Until someone bets, a player may check or bet 1 chip.
- Once someone has bet, each other player in turn may call for 1 chip or fold.
  There is no raising, so nobody acts more than twice.
- The hand ends when everyone has checked, or when everyone after the bettor
  has called or folded. The highest card among the players still in wins the pot.

Two standard games are defined here:
- KUHN_2P: the deck J Q K. 12 information sets. At equilibrium the first
  player loses 1/18 of a chip per hand.
- KUHN_3P: the deck J Q K A. 48 information sets. There is no single
  equilibrium value with three players: see Szafron, Gibson and Sturtevant,
  "A Parameterized Family of Equilibrium Profiles for Three-Player Kuhn Poker"
  (AAMAS 2013).

Other decks and player counts are made with KuhnRules.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import cache
from itertools import permutations

from pokerion.common.types import Action, InfoSetKey, Player
from pokerion.game.base import History

# The standard two-player deck, lowest card first. Kept as a module-level name
# because callers and tests import it.
CARDS = ["J", "Q", "K"]

# Characters that separate cards and actions inside deal strings and
# information set keys, so they cannot appear in a card name.
_RESERVED_CHARACTERS = ",|:"


@cache
def _rank_of(deck: tuple[str, ...]) -> dict[str, int]:
    return {card: rank for rank, card in enumerate(deck)}


@dataclass(frozen=True)
class KuhnRules:
    """Which Kuhn game is being played: how many players, and with which deck.

    `deck` lists the cards from lowest to highest. A card's position in the
    deck is its rank, so the names are only labels.
    """

    num_players: int = 2
    deck: tuple[str, ...] = ("J", "Q", "K")

    def __post_init__(self):
        if self.num_players < 2:
            raise ValueError("Kuhn poker needs at least two players")
        if len(self.deck) <= self.num_players:
            raise ValueError(
                f"{self.num_players} players need a deck of at least "
                f"{self.num_players + 1} cards, got {len(self.deck)}"
            )
        if len(set(self.deck)) != len(self.deck):
            raise ValueError("card names must be unique")
        for card in self.deck:
            if not card or any(ch in _RESERVED_CHARACTERS for ch in card):
                raise ValueError(
                    f"card name {card!r} is empty or contains one of {_RESERVED_CHARACTERS!r}"
                )

    def root(self) -> KuhnHistory:
        """The start of a hand, before the cards are dealt."""
        return KuhnHistory(rules=self)

    def rank(self, card: str) -> int:
        """Strength of a card: 0 for the lowest card in the deck."""
        return _rank_of(self.deck)[card]

    def encode_deal(self, cards: tuple[str, ...]) -> Action:
        """The chance action that deals `cards`, one to each seat in seat order.

        Decks of one-character names give compact strings such as "KQ", which
        is the form already stored in saved replays. Longer names are joined
        with commas.
        """
        return self._deal_separator.join(cards)

    def decode_deal(self, action: Action) -> tuple[str, ...]:
        """Inverse of encode_deal."""
        if self._deal_separator:
            return tuple(action.split(self._deal_separator))
        return tuple(action)

    @property
    def _deal_separator(self) -> str:
        return "" if all(len(card) == 1 for card in self.deck) else ","


KUHN_2P = KuhnRules()
KUHN_3P = KuhnRules(num_players=3, deck=("J", "Q", "K", "A"))


class KuhnHistory(History):
    """Immutable game tree node for Kuhn poker.

    cards:   the card dealt to each seat; empty before the deal
    actions: the betting actions so far ("check", "bet", "call", "fold")
    rules:   which Kuhn game this is; the standard two-player game by default

    Players act in a fixed cycle and nobody is ever skipped, so the k-th
    betting action is always taken by seat ``k % num_players``.
    """

    def __init__(
        self,
        cards: tuple[str, ...] = (),
        actions: tuple[str, ...] = (),
        rules: KuhnRules = KUHN_2P,
    ):
        self._cards = cards
        self._actions = actions
        self._rules = rules
        # Position of the bet in the action sequence, or None while everyone
        # has checked. Everything before it is a check; everything after it is
        # a call or a fold.
        self._bet_at = actions.index("bet") if "bet" in actions else None

    @property
    def rules(self) -> KuhnRules:
        return self._rules

    @property
    def variant(self) -> str:
        if self._rules == KUHN_2P:
            return "kuhn"
        return f"kuhn-{self._rules.num_players}p-{len(self._rules.deck)}c"

    @property
    def num_players(self) -> int:
        return self._rules.num_players

    # ---------------------------------------------------------------- chance

    def is_chance(self) -> bool:
        return len(self._cards) < self._rules.num_players

    def chance_actions(self) -> list[tuple[Action, float]]:
        if not self.is_chance():
            return []
        # Every seat is dealt at once: one outcome per ordered choice of cards.
        deals = list(permutations(self._rules.deck, self._rules.num_players))
        prob = 1.0 / len(deals)
        return [(self._rules.encode_deal(deal), prob) for deal in deals]

    def sample_chance(self) -> Action:
        if not self.is_chance():
            return ""
        deal = random.sample(self._rules.deck, self._rules.num_players)
        return self._rules.encode_deal(tuple(deal))

    # --------------------------------------------------------------- betting

    def is_terminal(self) -> bool:
        if self.is_chance():
            return False
        n = self._rules.num_players
        if self._bet_at is None:
            return len(self._actions) == n  # everyone checked
        # After a bet, each of the other n - 1 players answers exactly once.
        return len(self._actions) - self._bet_at - 1 == n - 1

    def active_player(self) -> Player:
        return len(self._actions) % self._rules.num_players

    def actions(self) -> list[Action]:
        if self.is_chance() or self.is_terminal():
            return []
        if self._bet_at is None:
            return ["check", "bet"]
        return ["call", "fold"]

    def _chips_and_live_seats(self) -> tuple[list[int], list[bool]]:
        """Chips each seat has put in, and which seats have not folded."""
        n = self._rules.num_players
        chips = [1] * n  # the antes
        live = [True] * n
        for k, action in enumerate(self._actions):
            seat = k % n
            if action in ("bet", "call"):
                chips[seat] += 1
            elif action == "fold":
                live[seat] = False
        return chips, live

    def _winner(self, live: list[bool]) -> Player:
        """The seat holding the highest card among those still in."""
        return max(
            (seat for seat, still_in in enumerate(live) if still_in),
            key=lambda seat: self._rules.rank(self._cards[seat]),
        )

    def terminal_utility(self, player: Player) -> float:
        if not self.is_terminal():
            raise ValueError(f"Not terminal: {self._actions}")
        chips, live = self._chips_and_live_seats()
        if player == self._winner(live):
            return float(sum(chips) - chips[player])  # the pot, less what they paid in
        return float(-chips[player])

    def showdown_players(self) -> list[Player]:
        if not self.is_terminal():
            raise ValueError(f"Not terminal: {self._actions}")
        _, live = self._chips_and_live_seats()
        still_in = [seat for seat, in_hand in enumerate(live) if in_hand]
        # With one player left there is nothing to compare, so no card is shown.
        return still_in if len(still_in) > 1 else []

    # ------------------------------------------------------------- knowledge

    def info_set_key(self) -> InfoSetKey:
        card = self._cards[self.active_player()]
        if not self._actions:
            return card
        return f"{card}|{':'.join(self._actions)}"

    def __add__(self, action: Action) -> KuhnHistory:
        if self.is_chance():
            return KuhnHistory(cards=self._rules.decode_deal(action), rules=self._rules)
        return KuhnHistory(
            cards=self._cards, actions=self._actions + (action,), rules=self._rules
        )

    # ------------------------------------------------------------------ views

    def to_state_dict(self, viewer: Player | None = None) -> dict:
        n = self._rules.num_players
        dealt = not self.is_chance()
        terminal = self.is_terminal()
        chips, live = self._chips_and_live_seats()

        players = []
        for seat in range(n):
            card = None
            if dealt:
                visible = viewer is None or viewer == seat or terminal
                card = self._cards[seat] if visible else "?"
            players.append({
                "id": seat,
                "card": card,
                # Chips this player has put into the pot this hand: ante + bets.
                # Stacks across hands are match-level state and are injected by
                # the Match, never invented here — this node must stay pure for
                # the solver's sake.
                "committed": chips[seat],
            })

        action_history = [
            {"player": k % n, "action": action} for k, action in enumerate(self._actions)
        ]
        in_play = dealt and not terminal

        return {
            "variant": self.variant,
            "viewer": viewer,  # whose eyes this view is through; None = god mode
            "players": players,
            "pot": sum(chips),
            "community_cards": [],
            "action_history": action_history,
            "legal_actions": self.actions() if in_play else [],
            "current_player": self.active_player() if in_play else None,
            "is_terminal": terminal,
            "is_chance": not dealt,
            "winner": self._winner(live) if terminal else None,
            "round": 0,
        }

    def __repr__(self) -> str:
        separator = self._rules._deal_separator
        cards = separator.join(self._cards) if self._cards else "??"
        actions = ",".join(self._actions) if self._actions else "-"
        return f"Kuhn({cards}|{actions})"
