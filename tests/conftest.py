"""Fixtures shared by the lab's tests.

Building a tree is the slow part, so it is done once per test session.
"""

import pytest

from pokerion.game.kuhn import KUHN_2P, KUHN_3P
from pokerion.game.tree import GameTree


@pytest.fixture(scope="session")
def kuhn() -> GameTree:
    """Standard two-player Kuhn poker."""
    return GameTree(KUHN_2P.root())


@pytest.fixture(scope="session")
def kuhn3() -> GameTree:
    """Standard three-player Kuhn poker."""
    return GameTree(KUHN_3P.root())
