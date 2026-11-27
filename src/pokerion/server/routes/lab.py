"""Lab API — run a match between bots and return everything needed to watch it.

This is for looking at the lab's work in a browser on your own machine. It is
off unless the environment variable POKERION_LAB is set, which
``python -m pokerion`` does for you. The deployed site leaves it off: every
request here plays a whole match, and nothing limits how often it is asked.

One request, one match. The response holds four things the page shows:

- ``per_hand``: what each agent expected to win on every hand. This is the score.
- ``plays``: the hand that was dealt each time, for the agents to learn from.
- ``references``: for an adapting agent facing fixed bots, what it would have
  won without adapting, and the most anything could have won in its place.
- ``beliefs``: what an adapting agent expected each opponent to play, at
  intervals through the match, beside what that opponent really played.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from functools import cache
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pokerion.arena import (
    AdaptiveAgent,
    Agent,
    CountingMemory,
    Entrant,
    MixWithBestResponse,
    PlayedHand,
    PlayTheCore,
    best_reply_to_table,
    bots,
    evaluate_table,
    play_match,
)
from pokerion.game.kuhn import KUHN_2P, KUHN_3P, KuhnRules
from pokerion.game.tree import GameTree
from pokerion.solver.regret_models import RegretTable
from pokerion.solver.tree_cfr import TreeCFR

MAX_HANDS = 2000
CORE_ITERATIONS = 2000     # of CFR+; enough for a near-exact equilibrium in Kuhn
BELIEF_CHECKPOINTS = 40    # how many times through a match the beliefs are recorded
NEAR_TIE = 1e-3            # actions within this much of the best count as equal


def lab_enabled() -> bool:
    return os.environ.get("POKERION_LAB", "").lower() in ("1", "true")


def _require_lab() -> None:
    if not lab_enabled():
        raise HTTPException(status_code=404, detail="the lab is not enabled on this server")


router = APIRouter(prefix="/api/lab", dependencies=[Depends(_require_lab)])


# ------------------------------------------------------------------ the menu

GAMES: dict[str, tuple[str, KuhnRules]] = {
    "kuhn-2p": ("Kuhn poker, 2 players", KUHN_2P),
    "kuhn-3p": ("Kuhn poker, 3 players", KUHN_3P),
}

DEFAULT_TABLES = {
    "kuhn-2p": ["adaptive", "rock"],
    "kuhn-3p": ["adaptive", "rock", "maniac"],
}


@dataclass(frozen=True)
class SeatKind:
    label: str
    description: str
    adapts: bool = False


SEAT_KINDS: dict[str, SeatKind] = {
    "adaptive": SeatKind(
        "Adapting agent",
        "Solid play that counts what each opponent does and moves towards the best reply.",
        adapts=True,
    ),
    "solid": SeatKind("Solid play", "The strategy found by self-play. It never adapts."),
    "rock": SeatKind("Rock", "Never bets, and folds to any bet."),
    "maniac": SeatKind("Maniac", "Bets whenever it can and never folds."),
    "station": SeatKind("Calling station", "Never bets and never folds."),
    "random": SeatKind("Random", "Every legal action equally likely."),
    "near_solid": SeatKind(
        "Nearly solid", "Solid play with every probability moved by up to 0.2, then fixed."
    ),
}

TIE_BREAKS: dict[str, tuple[str, ...]] = {
    "bet": ("bet", "raise", "call"),
    "check": ("check", "fold"),
    "none": (),
}


class SeatRequest(BaseModel):
    kind: str
    # The next two only matter for the adapting agent.
    trust: float = Field(1.0, ge=0.0, le=1.0, description="share of hands played as the best reply")
    ties: Literal["bet", "check", "none"] = "bet"


class MatchRequest(BaseModel):
    game: str = "kuhn-3p"
    seats: list[SeatRequest]
    hands: int = Field(300, ge=1, le=MAX_HANDS)
    reveal: Literal["showdown", "all"] = "all"
    seed: int = 0


@cache
def _game(game_id: str) -> tuple[GameTree, np.ndarray]:
    """The compiled game and its solid core. Solved once, then kept."""
    tree = GameTree(GAMES[game_id][1].root())
    solver = TreeCFR(tree, RegretTable(tree, floored=True), linear_averaging=True)
    solver.train(CORE_ITERATIONS)
    core = solver.average_strategy()
    core.setflags(write=False)
    return tree, core


def _error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


# ------------------------------------------------------------------ endpoints


@router.get("/options")
def options() -> dict:
    """What can be chosen for a match. The page builds its form from this."""
    return {
        "games": [
            {
                "id": game_id,
                "name": name,
                "players": rules.num_players,
                "table": DEFAULT_TABLES[game_id],
            }
            for game_id, (name, rules) in GAMES.items()
        ],
        "kinds": [
            {"id": kind, "label": k.label, "description": k.description, "adapts": k.adapts}
            for kind, k in SEAT_KINDS.items()
        ],
        "max_hands": MAX_HANDS,
    }


@router.post("/match")
def run_match(request: MatchRequest):
    """Play one match and return what is needed to watch it."""
    if request.game not in GAMES:
        return _error(f"unknown game {request.game!r}", 404)
    tree, core = _game(request.game)
    if len(request.seats) != tree.num_players:
        return _error(f"{GAMES[request.game][0]} needs {tree.num_players} seats")
    unknown = [seat.kind for seat in request.seats if seat.kind not in SEAT_KINDS]
    if unknown:
        return _error(f"unknown kind of player {unknown[0]!r}")

    rng = np.random.default_rng(request.seed)
    agents = _build_table(tree, core, request.seats, rng)
    adapting = [i for i, agent in enumerate(agents) if isinstance(agent, AdaptiveAgent)]

    # Beliefs are recorded before the first hand and then at even intervals.
    marks = sorted(set(np.linspace(0, request.hands, BELIEF_CHECKPOINTS + 1).astype(int).tolist()))
    snapshots: dict[int, dict[int, list]] = {
        i: {j: [_rows(tree, agents[i].predicted(j))] for j in range(len(agents)) if j != i}
        for i in adapting
    }
    plays: list[dict] = []

    def watch(hand: PlayedHand) -> None:
        plays.append(_describe(tree, hand, request.reveal))
        if hand.index + 1 in marks:
            for i in adapting:
                for j, history in snapshots[i].items():
                    history.append(_rows(tree, agents[i].predicted(j)))

    result = play_match(tree, agents, request.hands, rng, reveal=request.reveal, on_hand=watch)

    return {
        "game": {
            "id": request.game,
            "name": GAMES[request.game][0],
            "players": tree.num_players,
            "infosets": [
                {
                    "key": tree.infoset_keys[row],
                    "seat": int(tree.infoset_player[row]),
                    "actions": list(tree.infoset_actions[row]),
                }
                for row in range(tree.num_infosets)
            ],
        },
        "agents": [
            {"name": agent.name, "kind": seat.kind, "adapts": SEAT_KINDS[seat.kind].adapts}
            for agent, seat in zip(agents, request.seats, strict=True)
        ],
        "reveal": request.reveal,
        "hands": request.hands,
        "per_hand": np.round(result.per_hand, 5).tolist(),
        "winnings": np.round(result.winnings, 5).tolist(),
        "plays": plays,
        "references": _references(tree, core, agents, adapting),
        "beliefs": {
            str(i): {
                "hands_seen": marks,
                "about": {
                    str(j): {"predicted": history, "actual": _actual(tree, agents[j])}
                    for j, history in snapshots[i].items()
                },
            }
            for i in adapting
        },
    }


# ------------------------------------------------------------------- helpers


def _build_table(
    tree: GameTree, core: np.ndarray, seats: list[SeatRequest], rng: np.random.Generator
) -> list[Agent]:
    """One agent per requested seat. Two of a kind get numbered names."""
    totals = Counter(seat.kind for seat in seats)
    seen: Counter[str] = Counter()
    agents: list[Agent] = []
    for seat in seats:
        seen[seat.kind] += 1
        name = SEAT_KINDS[seat.kind].label
        if totals[seat.kind] > 1:
            name = f"{name} {seen[seat.kind]}"
        agents.append(_make_agent(tree, core, seat, name, rng))
    return agents


def _make_agent(
    tree: GameTree, core: np.ndarray, seat: SeatRequest, name: str, rng: np.random.Generator
) -> Agent:
    if seat.kind == "adaptive":
        rule = (
            MixWithBestResponse(seat.trust, prefer=TIE_BREAKS[seat.ties], tolerance=NEAR_TIE)
            if seat.trust > 0
            else PlayTheCore()
        )
        return AdaptiveAgent(name, core, CountingMemory(core), rule)
    fixed = {
        "solid": lambda: core,
        "rock": lambda: bots.rock(tree),
        "maniac": lambda: bots.maniac(tree),
        "station": lambda: bots.calling_station(tree),
        "random": lambda: bots.uniform_random(tree),
        "near_solid": lambda: bots.perturbed(tree, core, 0.2, rng),
    }
    return Entrant(name, fixed[seat.kind]())


def _describe(tree: GameTree, hand: PlayedHand, reveal: str) -> dict:
    """One dealt hand in the form the page draws: who sat where, the cards, the betting.

    ``showdown`` lists the seats still in at the end, if there was a showdown.
    ``shown`` lists the seats whose cards the other players got to see, which
    is everyone when the match turns every card over.
    """
    ending = int(tree.edge_child[hand.played[-1]])
    state = tree.replay(hand.played).to_state_dict()
    showdown = np.flatnonzero(tree.showdown[ending]).tolist()
    return {
        "seating": list(hand.seating),
        "cards": [player["card"] for player in state["players"]],
        "committed": [player["committed"] for player in state["players"]],
        "actions": [[step["player"], step["action"]] for step in state["action_history"]],
        "pot": state["pot"],
        "winner": state["winner"],
        "showdown": showdown,
        "shown": list(range(tree.num_players)) if reveal == "all" else showdown,
        "expected": np.round(hand.expected, 5).tolist(),
    }


def _rows(tree: GameTree, strategy: np.ndarray) -> list[list[float]]:
    """A strategy as one short list per information set: its legal actions, in the game's order."""
    return [
        [round(float(strategy[row, tree.column(action)]), 3) for action in actions]
        for row, actions in enumerate(tree.infoset_actions)
    ]


def _actual(tree: GameTree, agent: Agent) -> list[list[float]] | None:
    """What a fixed bot really plays. None for an agent that changes."""
    return _rows(tree, agent.strategy) if isinstance(agent, Entrant) else None


def _references(
    tree: GameTree, core: np.ndarray, agents: list[Agent], adapting: list[int]
) -> dict[str, dict[str, float]]:
    """For each adapting agent: its result had it not adapted, and the most possible.

    Both only mean something when everyone else at the table is fixed, so an
    adapting agent facing another adapting agent gets no references.
    """
    references = {}
    for hero in adapting:
        others = [agent for i, agent in enumerate(agents) if i != hero]
        if not all(isinstance(agent, Entrant) for agent in others):
            continue
        table = [Entrant("core", core) if i == hero else agent for i, agent in enumerate(agents)]
        references[str(hero)] = {
            "fixed": round(float(evaluate_table(tree, table).winnings[hero]), 5),
            "ceiling": round(best_reply_to_table(tree, table, hero), 5),
        }
    return references
