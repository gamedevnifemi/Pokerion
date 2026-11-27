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


def random_strategy(tree: GameTree, rng: np.random.Generator) -> np.ndarray:
    """A strategy drawn at random: at each information set, every way of
    spreading probability over the legal actions is equally likely.

    With two legal actions the probability of the first is uniform between 0
    and 1. This is a different bot on every call, unlike `uniform_random`,
    which is the single strategy that plays every action equally often.
    """
    draws = np.where(tree.legal, rng.exponential(size=tree.legal.shape), 0.0)
    return draws / draws.sum(axis=1, keepdims=True)


def perturbed(
    tree: GameTree, strategy: np.ndarray, spread: float, rng: np.random.Generator
) -> np.ndarray:
    """`strategy` with every action probability redrawn close to where it was.

    At an information set with two actions, the first probability is drawn
    uniformly from the values within `spread` of the original that are still
    probabilities, and the second is whatever remains. So an action played
    never, with a spread of 0.2, ends up played between 0 and 20% of the time.
    With more actions each probability is drawn that way and the row is then
    rescaled to sum to one.

    ``perturbed(tree, solid, 0.2, rng)`` is a bot that plays close to solid but
    not quite. It is how Ganzfried and Sandholm (2015) build their
    "sophisticated static" opponents.
    """
    if spread < 0:
        raise ValueError(f"spread must not be negative, got {spread}")
    low = np.clip(strategy - spread, 0.0, 1.0)
    high = np.clip(strategy + spread, 0.0, 1.0)
    shifted = np.where(tree.legal, rng.uniform(low, high), 0.0)

    two = tree.legal.sum(axis=1) == 2
    first = tree.legal.argmax(axis=1)  # column of the first legal action in each row
    rows = np.flatnonzero(two)
    kept = shifted[rows, first[rows]]
    shifted[rows] = np.where(tree.legal[rows], (1.0 - kept)[:, None], 0.0)
    shifted[rows, first[rows]] = kept

    total = shifted.sum(axis=1, keepdims=True)
    uniform = tree.uniform_strategy()
    return np.where(total > 0, shifted / np.where(total > 0, total, 1.0), uniform)


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
