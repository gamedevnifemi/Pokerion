"""Fixtures shared by the lab's tests.

Building a tree and solving a game are the slow parts, so each is done once per
test session. The arrays are made read-only so that no test can change what
another one sees.
"""

import numpy as np
import pytest

from pokerion.game.kuhn import KUHN_2P, KUHN_3P
from pokerion.game.tree import GameTree
from pokerion.solver.regret_models import RegretTable
from pokerion.solver.tree_cfr import TreeCFR


def solve_cfr_plus(tree: GameTree, iterations: int) -> np.ndarray:
    """The average strategy after `iterations` of CFR+, as a read-only array."""
    solver = TreeCFR(tree, RegretTable(tree, floored=True), linear_averaging=True)
    solver.train(iterations)
    strategy = solver.average_strategy()
    strategy.setflags(write=False)
    return strategy


@pytest.fixture(scope="session")
def kuhn() -> GameTree:
    """Standard two-player Kuhn poker."""
    return GameTree(KUHN_2P.root())


@pytest.fixture(scope="session")
def kuhn3() -> GameTree:
    """Standard three-player Kuhn poker."""
    return GameTree(KUHN_3P.root())


@pytest.fixture(scope="session")
def kuhn_equilibrium(kuhn) -> np.ndarray:
    """A near-exact equilibrium of two-player Kuhn."""
    return solve_cfr_plus(kuhn, 1000)


@pytest.fixture(scope="session")
def kuhn3_equilibrium(kuhn3) -> np.ndarray:
    """A near-exact equilibrium of three-player Kuhn."""
    return solve_cfr_plus(kuhn3, 1500)
