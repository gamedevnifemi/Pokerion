"""Matches: many hands at one table, between agents that may change how they play.

How a hand is scored
--------------------
Before each hand every agent says which strategy it will play. The hand is
then scored exactly: each agent is credited with what it expects to win under
those strategies, computed from the game tree. No cards decide the score.

One hand is also dealt and played out at random, and each agent is shown what
a player in its seat would have seen of it. That is what the agents learn
from. So the only luck left in a match is in what the agents get to observe,
never in who happens to win a pot.

What an agent is shown
----------------------
Every action is public. The information set an action was taken at is known
only if the cards behind it are known. An agent always knows its own. With
``reveal="showdown"`` it learns another player's only when the hand ends in a
showdown that player is part of. With ``reveal="all"`` every card is turned
over after every hand, which is the convention in the published Kuhn poker
experiments.

Agents are numbered by their position in the list given to `play_match`. The
numbers stay fixed for the match while the seats change from hand to hand.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import permutations
from typing import Protocol

import numpy as np

from pokerion.game.tree import CHANCE, GameTree

# The agent sitting in each seat, by its number at the table.
type Seating = tuple[int, ...]

REVEAL_MODES = ("showdown", "all")


@dataclass(frozen=True)
class Decision:
    """One betting action in a finished hand."""

    seat: int
    action: int      # column of the action taken
    row: int | None  # the information set it was taken at; None if those cards stayed hidden


@dataclass(frozen=True)
class HandView:
    """A finished hand, as seen from one seat."""

    seating: Seating
    seat: int                        # where the observer sat
    decisions: tuple[Decision, ...]  # every betting action, in order


class Agent(Protocol):
    """Something that can sit at a table for a match."""

    name: str

    def begin_match(self, tree: GameTree, me: int) -> None:
        """Forget any earlier match. `me` is this agent's number at the table."""
        ...

    def strategy_for(self, seating: Seating) -> np.ndarray:
        """The strategy it will play in the next hand, given who sits where.

        Only the rows of the agent's own seat are used.
        """
        ...

    def observe(self, hand: HandView) -> None:
        """Take in a finished hand, as seen from the agent's own seat."""
        ...


@dataclass(frozen=True)
class PlayedHand:
    """One hand of a match, for anything that is watching.

    `played` holds the edges of the hand that was dealt; pass it to
    `GameTree.replay` to see the cards and the betting. `expected` is what each
    seat expected to win before the cards were dealt, which is the hand's score.
    """

    index: int
    seating: Seating
    played: np.ndarray
    expected: np.ndarray  # (num_players,), by seat


@dataclass(frozen=True)
class MatchResult:
    """What each agent expected to win on each hand of a match."""

    names: tuple[str, ...]
    per_hand: np.ndarray  # (num_agents, hands), in chips

    @property
    def winnings(self) -> np.ndarray:
        """Chips per hand for each agent, averaged over the match."""
        return self.per_hand.mean(axis=1)


def play_match(
    tree: GameTree,
    agents: Sequence[Agent],
    hands: int,
    rng: np.random.Generator,
    *,
    reveal: str = "showdown",
    seatings: Sequence[Seating] | None = None,
    on_hand: Callable[[PlayedHand], None] | None = None,
) -> MatchResult:
    """Play `hands` hands between `agents` and score each one exactly.

    `seatings` lists the seatings to cycle through, one per hand. By default
    it is every arrangement of the agents, so that over a match each agent
    plays each seat equally often. Pass a single seating to keep everyone in
    place.

    `on_hand` is called after every hand, once the agents have taken it in. It
    is how a match is watched: nothing it does changes the match.
    """
    n = tree.num_players
    if len(agents) != n:
        raise ValueError(f"{tree.variant} seats {n} players, got {len(agents)} agents")
    if reveal not in REVEAL_MODES:
        raise ValueError(f"reveal must be one of {REVEAL_MODES}, got {reveal!r}")
    if seatings is None:
        schedule = list(permutations(range(n)))
    else:
        schedule = [tuple(seating) for seating in seatings]
    for seating in schedule:
        if sorted(seating) != list(range(n)):
            raise ValueError(f"seating {seating} does not place each of the {n} agents once")

    for number, agent in enumerate(agents):
        agent.begin_match(tree, number)

    per_hand = np.zeros((n, hands))
    for hand in range(hands):
        seating = schedule[hand % len(schedule)]
        profile = tree.combine([agents[number].strategy_for(seating) for number in seating])
        tree.check_strategy(profile)
        expected = tree.expected_utilities(profile)
        per_hand[list(seating), hand] = expected

        played = tree.sample_play(profile, rng)
        for view in _views(tree, played, seating, reveal):
            agents[seating[view.seat]].observe(view)
        if on_hand is not None:
            on_hand(PlayedHand(index=hand, seating=seating, played=played, expected=expected))

    return MatchResult(names=tuple(agent.name for agent in agents), per_hand=per_hand)


def play_matches(
    tree: GameTree,
    make_agents: Callable[[np.random.Generator], Sequence[Agent]],
    hands: int,
    matches: int,
    seed: int = 0,
    **options,
) -> np.ndarray:
    """Play independent matches and stack what each agent won on each hand.

    `make_agents` builds the table for one match. It is given that match's
    random generator, so it can draw a fresh random opponent each time.
    `options` are passed on to `play_match`.

    Returns an array of shape ``(matches, num_agents, hands)``. Averaging over
    the first axis gives the expected course of a match; its spread gives the
    uncertainty.
    """
    results = []
    for rng in np.random.default_rng(seed).spawn(matches):
        results.append(play_match(tree, make_agents(rng), hands, rng, **options).per_hand)
    return np.stack(results)


def _views(
    tree: GameTree, played: np.ndarray, seating: Seating, reveal: str
) -> list[HandView]:
    """What each seat saw of one played hand."""
    betting = played[tree.edge_owner[played] != CHANCE]
    seats = tree.edge_owner[betting].tolist()
    rows = (tree.edge_slot[betting] // tree.num_actions).tolist()
    actions = (tree.edge_slot[betting] % tree.num_actions).tolist()
    shown = tree.showdown[tree.edge_child[played[-1]]]

    views = []
    for observer in range(tree.num_players):
        decisions = tuple(
            Decision(
                seat=seat,
                action=action,
                row=row if (reveal == "all" or seat == observer or shown[seat]) else None,
            )
            for seat, row, action in zip(seats, rows, actions, strict=True)
        )
        views.append(HandView(seating=seating, seat=observer, decisions=decisions))
    return views
