"""Exact yardsticks for a strategy profile on a compiled game.

Everything here is computed from the full tree, so it carries no sampling
noise: two profiles that score differently really are different.

Terms
-----
Best response
    The strategy that earns one seat the most while everyone else keeps
    playing as they are.
Deviation gain
    How much a seat would gain by switching to its best response. When it is
    zero for every seat, the profile is a Nash equilibrium.
NashConv
    The deviation gains added up over all seats. In a two-player zero-sum game
    the literature's "exploitability" is NashConv divided by two. The web app's
    training view shows NashConv itself and calls it exploitability.

With three or more players a profile with zero NashConv is still an
equilibrium, but that no longer protects a seat against opponents who play
something else, so NashConv measures closeness to equilibrium and not safety.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pokerion.common.types import Action
from pokerion.game.tree import GameTree


@dataclass(frozen=True)
class BestResponse:
    """One seat's best response to a profile."""

    player: int
    value: float         # the seat's expected payoff per hand when it best-responds
    profile: np.ndarray  # the original profile with this seat's rows replaced by the best response


def best_response(
    tree: GameTree,
    profile: np.ndarray,
    player: int,
    *,
    prefer: Sequence[Action] = (),
    tolerance: float = 0.0,
) -> BestResponse:
    """The best `player` can do against the rest of `profile`.

    The response is deterministic: one action at each of the player's
    information sets. The player cannot tell apart the nodes of one
    information set, so the action there is chosen for all of them together,
    each node weighted by how likely chance and the other players make it.

    Ties
    ----
    Several actions are often equally good. Which of them is chosen does not
    change the value, but for an agent that learns from what it sees it decides
    what gets seen next, and that can change the course of a whole match.

    `prefer` names the actions to favour among equals, most favoured first.
    Actions it does not name come after, in column order. `tolerance` widens
    what counts as equal: at an information set, any action whose contribution
    to the player's expected payoff per hand is within `tolerance` of the best
    one may be chosen. With a positive tolerance the result is only close to a
    best response, and `value` is what it actually earns.

    Information sets the others never let the player reach are all ties.
    """
    if not tree.infosets_share_depth:
        raise NotImplementedError(
            "best_response needs every node of an information set to sit at the same depth"
        )
    if tolerance < 0:
        raise ValueError(f"tolerance must not be negative, got {tolerance}")

    # Lower rank wins a tie: the preferred actions first, then the rest by column.
    rank = len(prefer) + np.arange(tree.num_actions, dtype=np.float64)
    for position, action in enumerate(prefer):
        if action in tree.actions:
            rank[tree.column(action)] = position

    prob = tree.edge_probabilities(profile)
    weight = tree.reach(profile).without(player)
    value = tree.utilities[:, player].copy()
    choice = np.full(tree.num_infosets, -1, dtype=np.int64)
    num_slots = tree.num_infosets * tree.num_actions

    for level in reversed(tree.levels):
        parent, child = tree.edge_parent[level], tree.edge_child[level]
        mine = tree.edge_owner[level] == player

        # Chance and the other players: an ordinary expectation over their moves.
        others = ~mine
        value += np.bincount(
            parent[others],
            weights=prob[level][others] * value[child[others]],
            minlength=tree.num_nodes,
        )
        if not mine.any():
            continue

        # The player's own moves: score each action at each information set
        # over all the nodes it covers, then take the best action everywhere.
        slot = tree.edge_slot[level][mine]
        action_value = np.bincount(
            slot, weights=weight[parent[mine]] * value[child[mine]], minlength=num_slots
        ).reshape(tree.num_infosets, tree.num_actions)
        action_value[~tree.legal] = -np.inf
        rows = np.unique(slot // tree.num_actions)
        best = action_value[rows].max(axis=1, keepdims=True)
        as_good = action_value[rows] >= best - tolerance
        choice[rows] = np.where(as_good, rank, np.inf).argmin(axis=1)

        taken = slot % tree.num_actions == choice[slot // tree.num_actions]
        value[parent[mine][taken]] = value[child[mine][taken]]

    response = profile.copy()
    rows = np.flatnonzero(tree.infoset_player == player)
    response[rows] = 0.0
    response[rows, choice[rows]] = 1.0
    return BestResponse(player=player, value=float(value[0]), profile=response)


def deviation_gains(tree: GameTree, profile: np.ndarray) -> np.ndarray:
    """For each seat, what it would gain per hand by best-responding to the others."""
    current = tree.expected_utilities(profile)
    best = np.array([best_response(tree, profile, p).value for p in range(tree.num_players)])
    return best - current


def nash_conv(tree: GameTree, profile: np.ndarray) -> float:
    """The deviation gains added over all seats. Zero exactly at a Nash equilibrium."""
    return float(deviation_gains(tree, profile).sum())
