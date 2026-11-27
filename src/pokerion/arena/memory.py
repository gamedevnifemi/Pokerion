"""What an agent remembers about how each opponent has played.

A memory watches finished hands and, when asked, predicts the strategy of a
given opponent. Everything an agent does to exploit an opponent starts from
that prediction, which makes this the second place where methods compete:
any other memory is judged against `CountingMemory`.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from pokerion.arena.match import HandView
from pokerion.game.tree import GameTree


class OpponentMemory(Protocol):
    """What an `AdaptiveAgent` asks of its memory."""

    def begin_match(self, tree: GameTree, me: int) -> None:
        """Forget everything. `me` is the number of the agent this memory belongs to."""
        ...

    def observe(self, hand: HandView) -> None:
        """Take in a finished hand."""
        ...

    def predict(self, opponent: int) -> np.ndarray:
        """The strategy this opponent is expected to play, for every seat it might sit in."""
        ...


class CountingMemory:
    """Counts each opponent's actions at each information set, on top of a prior.

    At an information set the prediction is::

        (prior_weight * prior + counts) / (prior_weight + visits)

    It starts at the prior and moves towards the observed frequencies as the
    visits add up. ``prior_weight`` is how many imaginary visits the prior is
    worth. Five, with an equilibrium strategy as the prior, is the setting used
    by Ganzfried and Sandholm in "Safe Opponent Exploitation" (2015).

    Limitation: an action can only be counted if the information set it was
    taken at was shown. A hand that ends without a showdown therefore teaches
    this memory nothing about the players involved. Under
    ``reveal="showdown"`` that makes it blind to folds, and what it does learn
    is skewed towards hands that were played to the end.
    """

    def __init__(self, prior: np.ndarray, prior_weight: float = 5.0):
        if prior_weight <= 0:
            raise ValueError(
                "prior_weight must be positive, or unseen information sets have no prediction"
            )
        self._prior = prior
        self._prior_weight = prior_weight
        self._me = -1
        self._counts: dict[int, np.ndarray] = {}

    def begin_match(self, tree: GameTree, me: int) -> None:
        tree.check_strategy(self._prior)
        self._me = me
        self._counts = {}

    def observe(self, hand: HandView) -> None:
        for decision in hand.decisions:
            player = hand.seating[decision.seat]
            if player == self._me or decision.row is None:
                continue
            counts = self._counts.get(player)
            if counts is None:
                counts = self._counts[player] = np.zeros_like(self._prior)
            counts[decision.row, decision.action] += 1.0

    def predict(self, opponent: int) -> np.ndarray:
        counts = self._counts.get(opponent)
        if counts is None:
            return self._prior
        visits = counts.sum(axis=1, keepdims=True)
        return (self._prior_weight * self._prior + counts) / (self._prior_weight + visits)
