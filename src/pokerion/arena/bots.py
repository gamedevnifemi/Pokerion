"""Simple fixed strategies to seat at a table.

Each function returns a strategy array for a compiled game. The bots are
described by action names, so they work in any game that uses the same names:
"check", "bet", "call", "raise" and "fold". An action name a game does not
have is simply never legal there.

These are the weak opponents an adapting agent should profit from, and the
yardsticks a solid strategy is checked against.
"""

from __future__ import annotations

import numpy as np

from pokerion.common.types import Action
from pokerion.game.tree import GameTree


def uniform_random(tree: GameTree) -> np.ndarray:
    """Every legal action equally likely, everywhere."""
    return tree.uniform_strategy()


def prefer(tree: GameTree, *preferences: Action) -> np.ndarray:
    """Always take the first action in `preferences` that is legal.

    Where none of them is legal, play uniformly over what is.
    """
    strategy = np.zeros((tree.num_infosets, tree.num_actions))
    undecided = np.ones(tree.num_infosets, dtype=bool)
    for action in preferences:
        if action not in tree.actions:
            continue
        column = tree.column(action)
        takes_it = undecided & tree.legal[:, column]
        strategy[takes_it, column] = 1.0
        undecided &= ~takes_it
    strategy[undecided] = tree.uniform_strategy()[undecided]
    return strategy


def calling_station(tree: GameTree) -> np.ndarray:
    """Never bets and never folds: it checks, and calls whatever it faces."""
    return prefer(tree, "call", "check")


def maniac(tree: GameTree) -> np.ndarray:
    """Bets or raises whenever it can, and calls when it cannot."""
    return prefer(tree, "bet", "raise", "call")


def rock(tree: GameTree) -> np.ndarray:
    """Never puts in a chip it does not have to: it checks, and folds to any bet."""
    return prefer(tree, "check", "fold")


def blend(first: np.ndarray, second: np.ndarray, weight: float) -> np.ndarray:
    """Mix two strategies at every information set.

    ``weight`` is the share of `second`: 0 gives `first`, 1 gives `second`.
    ``blend(solid, uniform_random(tree), 0.2)`` is solid play with noise.

    The mix is made separately at each decision. That is not the same as
    choosing one of the two strategies for a whole hand.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be between 0 and 1, got {weight}")
    return (1.0 - weight) * first + weight * second
