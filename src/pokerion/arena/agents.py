"""Agents that change how they play as a match goes on."""

from __future__ import annotations

import numpy as np

from pokerion.arena.exploitation import ExploitationRule
from pokerion.arena.match import HandView, Seating
from pokerion.arena.memory import OpponentMemory
from pokerion.game.tree import GameTree


class AdaptiveAgent:
    """An agent assembled from three parts.

    core
        The solid strategy it plays when it knows nothing about the table.
    memory
        What it has learned about each opponent, as a predicted strategy.
    rule
        How the predictions move its play away from the core.

    Each part can be swapped on its own, which is how one method is compared
    with another. With the rule `PlayTheCore` the agent never adapts, and is
    the reference the adapting versions are measured against.
    """

    def __init__(
        self, name: str, core: np.ndarray, memory: OpponentMemory, rule: ExploitationRule
    ):
        self.name = name
        self._core = core
        self._memory = memory
        self._rule = rule
        self._tree: GameTree | None = None
        self._me = -1

    def begin_match(self, tree: GameTree, me: int) -> None:
        tree.check_strategy(self._core)
        self._tree = tree
        self._me = me
        self._memory.begin_match(tree, me)

    def strategy_for(self, seating: Seating) -> np.ndarray:
        tree = self._tree
        if tree is None:
            raise RuntimeError("begin_match must be called before strategy_for")
        # Seat the predictions where the opponents will sit. The agent's own
        # seat holds the core only as a placeholder: the rule decides what goes there.
        table = tree.combine(
            [
                self._core if number == self._me else self._memory.predict(number)
                for number in seating
            ]
        )
        return self._rule.choose(tree, self._core, table, seating.index(self._me))

    def observe(self, hand: HandView) -> None:
        self._memory.observe(hand)

    def predicted(self, opponent: int) -> np.ndarray:
        """What the agent currently expects this opponent to play. For looking, not for play."""
        return self._memory.predict(opponent)
