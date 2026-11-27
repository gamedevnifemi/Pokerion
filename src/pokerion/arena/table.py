"""A table of bots, scored exactly.

When nobody at the table changes how they play, what each of them wins on
average can be computed from the game tree: no hands are dealt, so luck plays
no part in the result. This module does that for fixed strategies.

Seats matter in poker, so every result is averaged over all the ways the
entrants can be seated.

Matches between agents that adapt as they play are not here yet. They will
reuse this scoring, hand by hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import permutations

import numpy as np

from pokerion.evaluation import best_response
from pokerion.game.tree import GameTree


@dataclass(frozen=True)
class Entrant:
    """A named fixed strategy. It plays whichever seat it is given."""

    name: str
    strategy: np.ndarray


@dataclass(frozen=True)
class TableResult:
    """What each entrant wins per hand, in expectation.

    Rows follow the order the entrants were given in, so two entrants may
    share a name.
    """

    names: tuple[str, ...]
    by_seat: np.ndarray  # (num_entrants, num_players): chips per hand when sitting in each seat

    @property
    def winnings(self) -> np.ndarray:
        """Chips per hand for each entrant, averaged over the seats."""
        return self.by_seat.mean(axis=1)


def _seatings(num_players: int) -> list[tuple[int, ...]]:
    """Every seating: ``seating[seat]`` is the index of the entrant sitting there."""
    return list(permutations(range(num_players)))


def _check_table(tree: GameTree, entrants: Sequence[Entrant]) -> None:
    if len(entrants) != tree.num_players:
        raise ValueError(
            f"{tree.variant} seats {tree.num_players} players, got {len(entrants)} entrants"
        )
    for entrant in entrants:
        tree.check_strategy(entrant.strategy)


def evaluate_table(tree: GameTree, entrants: Sequence[Entrant]) -> TableResult:
    """Expected chips per hand for each entrant, over every seating.

    Costs one pass over the tree per seating, and there are ``num_players!``
    seatings: fine for the small tables this is meant for.
    """
    _check_table(tree, entrants)
    n = tree.num_players
    seatings = _seatings(n)
    totals = np.zeros((n, n))
    for seating in seatings:
        profile = tree.combine([entrants[i].strategy for i in seating])
        payoff = tree.expected_utilities(profile)
        for seat, entrant_index in enumerate(seating):
            totals[entrant_index, seat] += payoff[seat]
    # Each entrant sits in each seat in (n - 1)! of the n! seatings.
    by_seat = totals / (len(seatings) / n)
    return TableResult(names=tuple(e.name for e in entrants), by_seat=by_seat)


def best_reply_to_table(tree: GameTree, entrants: Sequence[Entrant], hero: int) -> float:
    """The most that the entrant at index `hero` could win per hand at this table.

    The other entrants keep their strategies. In each seating the hero is
    replaced by its best response to the players it faces there. The result is
    the ceiling for anything sitting in the hero's place, and the hero's actual
    winnings can be read as a share of it.
    """
    _check_table(tree, entrants)
    seatings = _seatings(tree.num_players)
    total = 0.0
    for seating in seatings:
        profile = tree.combine([entrants[i].strategy for i in seating])
        total += best_response(tree, profile, seating.index(hero)).value
    return total / len(seatings)


def table_mix_curve(
    tree: GameTree, hero: Entrant, strong: Entrant, weak: Entrant
) -> np.ndarray:
    """The hero's chips per hand as weak opponents replace strong ones.

    Entry ``k`` is the hero's result with ``k`` weak opponents and the rest
    strong, for ``k`` from 0 to ``num_players - 1``.
    """
    opponents = tree.num_players - 1
    curve = np.empty(opponents + 1)
    for weak_count in range(opponents + 1):
        table = [hero] + [weak] * weak_count + [strong] * (opponents - weak_count)
        curve[weak_count] = evaluate_table(tree, table).winnings[0]
    return curve
