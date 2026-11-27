"""Shared test helpers: a tiny game with a known answer, and slow reference calculations.

The reference calculations walk the `History` tree directly and share no code
with the array engine, so agreement between the two is real evidence.
"""

from __future__ import annotations

import random

import numpy as np

from pokerion.common.types import Action, InfoSetKey, Player
from pokerion.game.base import History
from pokerion.game.tree import GameTree


class BiasedCoin(History):
    """Chance flips a coin that lands heads 70% of the time. Player 0 guesses it blind.

    A correct guess wins one chip from player 1, who has no moves. Guessing
    heads with probability q earns player 0 exactly 0.4 * (2q - 1), so the best
    reply is always heads, worth 0.4.

    The point of the game is that chance is not uniform. An algorithm that
    forgets to weight nodes by their chance probability sees heads and tails as
    equally good guesses and gets this wrong.

    With ``lopsided=True`` the legal actions depend on the coin, which the
    player cannot see: a broken game that the tree builder must reject.
    """

    def __init__(self, coin: str | None = None, guess: str | None = None, lopsided: bool = False):
        self._coin = coin
        self._guess = guess
        self._lopsided = lopsided

    @property
    def variant(self) -> str:
        return "biased-coin"

    @property
    def num_players(self) -> int:
        return 2

    def is_chance(self) -> bool:
        return self._coin is None

    def chance_actions(self) -> list[tuple[Action, float]]:
        return [("H", 0.7), ("T", 0.3)]

    def sample_chance(self) -> Action:
        return random.choices(["H", "T"], weights=[0.7, 0.3])[0]

    def is_terminal(self) -> bool:
        return self._guess is not None

    def terminal_utility(self, player: Player) -> float:
        won = self._guess.upper() == self._coin
        return (1.0 if won else -1.0) * (1.0 if player == 0 else -1.0)

    def showdown_players(self) -> list[Player]:
        return []

    def active_player(self) -> Player:
        return 0

    def actions(self) -> list[Action]:
        if self._lopsided and self._coin == "T":
            return ["h"]
        return ["h", "t"]

    def info_set_key(self) -> InfoSetKey:
        return "guess"

    def __add__(self, action: Action) -> BiasedCoin:
        if self.is_chance():
            return BiasedCoin(coin=action, lopsided=self._lopsided)
        return BiasedCoin(coin=self._coin, guess=action, lopsided=self._lopsided)

    def to_state_dict(self, viewer: Player | None = None) -> dict:
        return {"variant": self.variant}


def walk_expected_utilities(
    root: History, strategy: dict[InfoSetKey, dict[Action, float]]
) -> list[float]:
    """Each seat's expected payoff, by plain recursion over the game."""

    def value(history: History) -> list[float]:
        if history.is_terminal():
            return [history.terminal_utility(p) for p in range(history.num_players)]
        if history.is_chance():
            moves = history.chance_actions()
        else:
            probs = strategy[history.info_set_key()]
            moves = [(action, probs[action]) for action in history.actions()]
        total = [0.0] * history.num_players
        for action, prob in moves:
            for seat, v in enumerate(value(history + action)):
                total[seat] += prob * v
        return total

    return value(root)


def random_strategy(tree: GameTree, rng: np.random.Generator) -> np.ndarray:
    """A random valid strategy: an independent random distribution at every information set."""
    weights = np.where(tree.legal, rng.random((tree.num_infosets, tree.num_actions)) + 0.05, 0.0)
    return weights / weights.sum(axis=1, keepdims=True)


def random_pure_strategy(tree: GameTree, rng: np.random.Generator) -> np.ndarray:
    """A random deterministic strategy: one legal action at every information set."""
    scores = np.where(tree.legal, rng.random((tree.num_infosets, tree.num_actions)), -1.0)
    pure = np.zeros_like(scores)
    pure[np.arange(tree.num_infosets), scores.argmax(axis=1)] = 1.0
    return pure
