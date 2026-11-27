"""Where a solver keeps what it has learned about regret.

Counterfactual regret minimisation needs two things from its memory: the
current estimate of every cumulative regret, and a way to fold in the regrets
measured this iteration. `RegretModel` is that contract. Keeping it separate
from the solver means a compact model can stand in for the table without the
solver changing, and the two can be compared like for like.

`RegretTable` is the exact reference. Anything else is judged against it.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from pokerion.game.tree import GameTree


class RegretModel(Protocol):
    """What `TreeCFR` asks of its regret memory."""

    def estimates(self) -> np.ndarray:
        """Current cumulative regrets, shape ``(num_infosets, num_actions)``."""
        ...

    def update(self, regrets: np.ndarray) -> None:
        """Fold in the regrets measured this pass.

        `regrets` has the same shape as `estimates`. Rows of seats that were
        not updated in this pass are zero.
        """
        ...


class RegretTable:
    """One number per information set and action: the exact running total.

    With ``floored=True`` each total is kept at or above zero after every
    update. This is regret matching+ (Tammelin, "Solving Large Imperfect
    Information Games Using CFR+", 2014): an action that has been bad for a long
    time can come back as soon as it does well, instead of first having to pay
    off the negative regret it built up.
    """

    def __init__(self, tree: GameTree, *, floored: bool = False):
        self.floored = floored
        self._total = np.zeros((tree.num_infosets, tree.num_actions))

    def estimates(self) -> np.ndarray:
        return self._total

    def update(self, regrets: np.ndarray) -> None:
        self._total = self._total + regrets
        if self.floored:
            np.maximum(self._total, 0.0, out=self._total)
