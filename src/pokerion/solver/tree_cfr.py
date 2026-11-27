"""Counterfactual regret minimisation on a compiled game tree.

This is the solver the lab uses for exact work. It does the same job as
`VanillaCFR`, but on arrays, so an iteration is a handful of numpy calls
instead of a Python recursion. It also leaves the storage of regrets to a
`RegretModel`, so that other ways of remembering regret can be compared
against the table.

It works for any number of players. In a two-player zero-sum game the average
strategy converges to a Nash equilibrium. With three or more players there is
no such guarantee, although in poker-like games the result is usually strong.
Measure it with `pokerion.evaluation`.

How an iteration works
----------------------
1. Turn the current regret estimates into a strategy by regret matching.
2. For each seat, measure the counterfactual regret of every action: how much
   better the action would have done than the strategy did, weighted by how
   likely chance and the other players make that situation.
3. Hand those regrets to the regret model, and add the strategy to the running
   average, weighted by how likely the seat itself is to reach each situation.

The strategy to use afterwards is the average, not the last one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from pokerion.game.tree import GameTree
from pokerion.solver.regret_models import RegretModel, RegretTable


def regret_matching(regrets: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """Play each legal action in proportion to its positive regret.

    Where no legal action has positive regret, play uniformly over the legal ones.
    """
    positive = np.where(legal, np.maximum(regrets, 0.0), 0.0)
    total = positive.sum(axis=1, keepdims=True)
    has_positive = total > 0
    uniform = legal / legal.sum(axis=1, keepdims=True)
    return np.where(has_positive, positive / np.where(has_positive, total, 1.0), uniform)


class TreeCFR:
    """CFR on a `GameTree`.

    Parameters
    ----------
    tree
        The compiled game.
    regrets
        Where regrets are kept. Defaults to an exact `RegretTable`.
    alternating
        If true, seats are updated one after another within an iteration, each
        seeing the strategies the earlier seats have just moved to. If false,
        all seats are updated at once from the same strategies. Alternating
        usually converges faster.
    linear_averaging
        If true, iteration ``t`` counts ``t`` times in the average strategy, so
        early, poor iterations fade. If false, every iteration counts equally.

    ``RegretTable(tree, floored=True)`` with ``alternating=True`` and
    ``linear_averaging=True`` is CFR+.
    """

    def __init__(
        self,
        tree: GameTree,
        regrets: RegretModel | None = None,
        *,
        alternating: bool = True,
        linear_averaging: bool = False,
    ):
        self.tree = tree
        self.regrets: RegretModel = regrets if regrets is not None else RegretTable(tree)
        self.alternating = alternating
        self.linear_averaging = linear_averaging
        self.iterations = 0
        self._strategy_sum = np.zeros((tree.num_infosets, tree.num_actions))
        # The edges each seat chooses, looked up once.
        self._edges_of = [
            np.flatnonzero(tree.edge_owner == seat) for seat in range(tree.num_players)
        ]

    def train(self, iterations: int, callback: Callable[[int], None] | None = None) -> None:
        """Run `iterations` more iterations, calling ``callback(i)`` after each."""
        seats = range(self.tree.num_players)
        for _ in range(iterations):
            if self.alternating:
                for seat in seats:
                    self._update([seat])
            else:
                self._update(seats)
            self.iterations += 1
            if callback:
                callback(self.iterations)

    def current_strategy(self) -> np.ndarray:
        """The strategy the next iteration will play. It does not converge."""
        return regret_matching(self.regrets.estimates(), self.tree.legal)

    def average_strategy(self) -> np.ndarray:
        """The strategy to use: the reach-weighted average over all iterations."""
        total = self._strategy_sum.sum(axis=1, keepdims=True)
        seen = total > 0
        uniform = self.tree.uniform_strategy()
        return np.where(seen, self._strategy_sum / np.where(seen, total, 1.0), uniform)

    def _update(self, seats: Iterable[int]) -> None:
        """Measure regrets under the current strategy and learn from them, for `seats`."""
        tree = self.tree
        strategy = self.current_strategy()
        reach = tree.reach(strategy)
        values = tree.node_values(strategy)
        weight = self.iterations + 1 if self.linear_averaging else 1.0

        regrets = np.zeros_like(strategy)
        for seat in seats:
            edges = self._edges_of[seat]
            parent, child = tree.edge_parent[edges], tree.edge_child[edges]

            # How much better each action is than the strategy at that node,
            # weighted by how likely the others make the node.
            advantage = values[child, seat] - values[parent, seat]
            regrets += tree.sum_by_slot(edges, reach.without(seat)[parent] * advantage)

            # The seat's own reach decides how much this iteration's strategy
            # counts at each information set in the average.
            played = strategy.ravel()[tree.edge_slot[edges]]
            self._strategy_sum += weight * tree.sum_by_slot(
                edges, reach.players[seat, parent] * played
            )

        self.regrets.update(regrets)
