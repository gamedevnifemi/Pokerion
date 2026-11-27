"""A game compiled into arrays, for exact computation.

`History` describes a game as a tree you can walk. Walking it in Python is
fine for one traversal and far too slow for thousands, which is what exact
evaluation and regret minimisation need. `GameTree` walks the game once and
stores it as flat numpy arrays. After that, reach probabilities, expected
winnings and node values are each a few array operations.

It only suits games small enough to enumerate in full: Kuhn, Leduc and their
scaled variants.

Layout
------
Nodes are numbered breadth-first from the root, so a child always has a larger
number than its parent. Edges are stored in the same order, which makes every
depth level one contiguous slice of the edge arrays (`GameTree.levels`). A
pass down the tree visits the slices in order; a pass up visits them in
reverse.

Strategies
----------
An information set is a row and an action is a column. A strategy is a float
array of shape ``(num_infosets, num_actions)``: each row holds the probability
of each action at that information set, and actions that are not legal there
are zero.

Information sets of different players are different rows, so the same shape
holds one player's strategy or a whole profile. `GameTree.combine` builds a
profile from one array per seat. Everything in this module that takes a
`profile` reads every row; rows of players you do not care about still need
valid probabilities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from pokerion.common.types import Action, InfoSetKey
from pokerion.game.base import History

# Value of `edge_owner` for an edge chosen by chance instead of by a player.
CHANCE = -1


@dataclass(frozen=True)
class Reach:
    """How likely each node is to be reached, split by who is responsible.

    The probability of reaching a node is a product of one factor per player
    (the probabilities of that player's own actions on the way there) and one
    for chance. Keeping the factors apart is what makes counterfactual values
    and best responses cheap: both need the product with one player left out.
    """

    players: np.ndarray  # (num_players, num_nodes)
    chance: np.ndarray   # (num_nodes,)

    def total(self) -> np.ndarray:
        """Probability of reaching each node when everyone follows the profile."""
        return self.chance * self.players.prod(axis=0)

    def without(self, player: int) -> np.ndarray:
        """Reach of each node counting chance and everyone except `player`.

        This is the weight a node carries in that player's counterfactual
        values: how likely the others make it, had the player tried to get there.
        """
        others = np.delete(self.players, player, axis=0)
        return self.chance * others.prod(axis=0)


class GameTree:
    """An extensive-form game stored as arrays. Immutable once built.

    Attributes
    ----------
    variant, num_players
        Copied from the root `History`.
    num_nodes, num_infosets, num_actions
        Sizes. ``num_actions`` counts every action name the game uses anywhere.
    actions
        The action names, in column order.
    infoset_keys, infoset_player, infoset_actions
        For each row: its key, the seat that acts there, and its legal actions
        in the game's own order.
    legal
        Boolean ``(num_infosets, num_actions)``: which actions each row allows.
    infosets_share_depth
        True when all the nodes of each information set sit at the same depth.
        `pokerion.evaluation.best_response` relies on it.
    utilities
        ``(num_nodes, num_players)``: payoffs at terminal nodes, zero elsewhere.
    edge_parent, edge_child
        The two ends of each edge.
    edge_owner
        Who chooses the edge: a seat number, or `CHANCE`.
    edge_chance_prob
        Probability of a chance edge; zero for a player's edge.
    edge_slot
        For a player's edge, ``row * num_actions + column``: where its
        probability sits in a flattened strategy. Zero for a chance edge.
    levels
        One slice of the edge arrays per depth, starting below the root.
    """

    def __init__(self, root: History, *, max_nodes: int = 2_000_000):
        """Enumerate the whole game below `root`.

        Raises ValueError if the game has more than `max_nodes` nodes, if a
        chance node's probabilities do not sum to one, or if one information
        set key is used for different players or different legal actions.
        """
        self.variant = root.variant
        self.num_players = root.num_players

        # Step 1: walk the game one depth at a time, collecting plain lists.
        row_of: dict[InfoSetKey, int] = {}
        column_of: dict[Action, int] = {}
        infoset_keys: list[InfoSetKey] = []
        infoset_player: list[int] = []
        infoset_actions: list[tuple[Action, ...]] = []
        infoset_depth: list[int] = []
        same_depth = True

        utilities: list[list[float]] = [[0.0] * self.num_players]
        parents: list[int] = []
        children: list[int] = []
        owners: list[int] = []
        chance_probs: list[float] = []
        rows: list[int] = []
        columns: list[int] = []
        levels: list[slice] = []

        frontier: list[tuple[int, History]] = [(0, root)]
        depth = 0
        while frontier:
            level_start = len(parents)
            next_frontier: list[tuple[int, History]] = []
            for node, history in frontier:
                if history.is_terminal():
                    utilities[node] = [
                        float(history.terminal_utility(p)) for p in range(self.num_players)
                    ]
                    continue

                if history.is_chance():
                    outcomes = history.chance_actions()
                    total = sum(prob for _, prob in outcomes)
                    if abs(total - 1.0) > 1e-9:
                        raise ValueError(f"chance probabilities sum to {total} at {history!r}")
                    moves = [(history + action, CHANCE, prob, 0, 0) for action, prob in outcomes]
                else:
                    player = history.active_player()
                    legal_here = tuple(history.actions())
                    if not legal_here:
                        raise ValueError(f"no legal actions at non-terminal node {history!r}")
                    key = history.info_set_key()
                    row = row_of.get(key)
                    if row is None:
                        row = row_of[key] = len(infoset_keys)
                        infoset_keys.append(key)
                        infoset_player.append(player)
                        infoset_actions.append(legal_here)
                        infoset_depth.append(depth)
                    elif infoset_player[row] != player or infoset_actions[row] != legal_here:
                        raise ValueError(
                            f"information set {key!r} is used for different players "
                            "or with different legal actions"
                        )
                    elif infoset_depth[row] != depth:
                        same_depth = False
                    moves = []
                    for action in legal_here:
                        column = column_of.setdefault(action, len(column_of))
                        moves.append((history + action, player, 0.0, row, column))

                for child_history, owner, prob, row, column in moves:
                    child = len(utilities)
                    if child >= max_nodes:
                        raise ValueError(
                            f"game has more than {max_nodes} nodes; it is too large to enumerate"
                        )
                    utilities.append([0.0] * self.num_players)
                    parents.append(node)
                    children.append(child)
                    owners.append(owner)
                    chance_probs.append(prob)
                    rows.append(row)
                    columns.append(column)
                    next_frontier.append((child, child_history))

            if len(parents) > level_start:
                levels.append(slice(level_start, len(parents)))
            frontier = next_frontier
            depth += 1

        # Step 2: freeze the lists into arrays.
        self.num_nodes = len(utilities)
        self.actions: tuple[Action, ...] = tuple(column_of)
        self.num_actions = len(self.actions)
        self.infoset_keys: tuple[InfoSetKey, ...] = tuple(infoset_keys)
        self.num_infosets = len(infoset_keys)
        self.infoset_player = np.array(infoset_player, dtype=np.int64)
        self.infoset_actions = tuple(infoset_actions)
        self.infosets_share_depth = same_depth

        self.legal = np.zeros((self.num_infosets, self.num_actions), dtype=bool)
        for row, legal_here in enumerate(infoset_actions):
            self.legal[row, [column_of[a] for a in legal_here]] = True

        self.utilities = np.array(utilities, dtype=np.float64)
        self.edge_parent = np.array(parents, dtype=np.int64)
        self.edge_child = np.array(children, dtype=np.int64)
        self.edge_owner = np.array(owners, dtype=np.int64)
        self.edge_chance_prob = np.array(chance_probs, dtype=np.float64)
        self.edge_slot = np.array(rows, dtype=np.int64) * self.num_actions + np.array(
            columns, dtype=np.int64
        )
        self.levels: tuple[slice, ...] = tuple(levels)

        self._row_of = row_of
        self._column_of = column_of
        self._is_player_edge = self.edge_owner != CHANCE

    # ------------------------------------------------------------ strategies

    def row(self, key: InfoSetKey) -> int:
        """Row of an information set."""
        return self._row_of[key]

    def column(self, action: Action) -> int:
        """Column of an action."""
        return self._column_of[action]

    def uniform_strategy(self) -> np.ndarray:
        """Every legal action equally likely, at every information set."""
        return self.legal / self.legal.sum(axis=1, keepdims=True)

    def strategy_from_dict(
        self, strategy: Mapping[InfoSetKey, Mapping[Action, float]]
    ) -> np.ndarray:
        """Convert the ``{key: {action: probability}}`` form into an array.

        Information sets missing from the mapping are played uniformly, which
        is also what the server does when a stored strategy has no entry.
        """
        array = self.uniform_strategy()
        for key, action_probs in strategy.items():
            row = self._row_of[key]
            array[row] = 0.0
            for action, prob in action_probs.items():
                array[row, self._column_of[action]] = prob
        self.check_strategy(array)
        return array

    def strategy_to_dict(self, strategy: np.ndarray) -> dict[InfoSetKey, dict[Action, float]]:
        """Convert an array into the ``{key: {action: probability}}`` form."""
        return {
            key: {a: float(strategy[row, self._column_of[a]]) for a in self.infoset_actions[row]}
            for row, key in enumerate(self.infoset_keys)
        }

    def check_strategy(self, strategy: np.ndarray, *, atol: float = 1e-8) -> None:
        """Raise ValueError unless every row is a distribution over its legal actions."""
        expected = (self.num_infosets, self.num_actions)
        if strategy.shape != expected:
            raise ValueError(f"strategy has shape {strategy.shape}, expected {expected}")
        if (strategy < -atol).any():
            raise ValueError("strategy has negative probabilities")
        if (np.abs(strategy[~self.legal]) > atol).any():
            raise ValueError("strategy puts probability on illegal actions")
        off = np.flatnonzero(np.abs(strategy.sum(axis=1) - 1.0) > atol)
        if off.size:
            raise ValueError(
                f"probabilities do not sum to 1 at information set {self.infoset_keys[off[0]]!r}"
            )

    def combine(self, strategies: Sequence[np.ndarray]) -> np.ndarray:
        """The profile in which seat ``p`` plays ``strategies[p]``.

        Each row is taken from the array of the seat that acts there.
        """
        if len(strategies) != self.num_players:
            raise ValueError(f"need {self.num_players} strategies, got {len(strategies)}")
        stacked = np.stack(strategies)
        return stacked[self.infoset_player, np.arange(self.num_infosets)]

    # ---------------------------------------------------------------- passes

    def edge_probabilities(self, profile: np.ndarray) -> np.ndarray:
        """Probability of taking each edge from its parent, under `profile`."""
        return np.where(
            self._is_player_edge, profile.ravel()[self.edge_slot], self.edge_chance_prob
        )

    def reach(self, profile: np.ndarray) -> Reach:
        """Reach probabilities of every node. One pass down the tree."""
        prob = self.edge_probabilities(profile)
        players = np.ones((self.num_players, self.num_nodes))
        chance = np.ones(self.num_nodes)
        for level in self.levels:
            parent, child = self.edge_parent[level], self.edge_child[level]
            owner, p = self.edge_owner[level], prob[level]
            chance[child] = chance[parent] * np.where(owner == CHANCE, p, 1.0)
            for seat in range(self.num_players):
                players[seat, child] = players[seat, parent] * np.where(owner == seat, p, 1.0)
        return Reach(players=players, chance=chance)

    def expected_utilities(self, profile: np.ndarray) -> np.ndarray:
        """Each seat's expected payoff per hand when everyone follows `profile`."""
        return self.reach(profile).total() @ self.utilities

    def node_values(self, profile: np.ndarray) -> np.ndarray:
        """Each seat's expected payoff from every node onward. One pass up the tree.

        Returns ``(num_nodes, num_players)``. Row 0 is the root, so it equals
        `expected_utilities`.
        """
        prob = self.edge_probabilities(profile)
        values = self.utilities.copy()
        for level in reversed(self.levels):
            parent, child = self.edge_parent[level], self.edge_child[level]
            weighted = prob[level][:, None] * values[child]
            for seat in range(self.num_players):
                values[:, seat] += np.bincount(
                    parent, weights=weighted[:, seat], minlength=self.num_nodes
                )
        return values

    def sum_by_slot(self, edges: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Add up one value per edge into an ``(num_infosets, num_actions)`` array.

        `edges` selects player edges (a boolean mask or an index array over the
        edge arrays) and `values` gives one number for each selected edge. Every
        edge that leaves the same information set by the same action lands in
        the same cell.
        """
        size = self.num_infosets * self.num_actions
        summed = np.bincount(self.edge_slot[edges], weights=values, minlength=size)
        return summed.reshape(self.num_infosets, self.num_actions)

    def __repr__(self) -> str:
        return (
            f"GameTree({self.variant!r}, players={self.num_players}, nodes={self.num_nodes}, "
            f"infosets={self.num_infosets})"
        )
