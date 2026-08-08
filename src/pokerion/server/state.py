"""Match model and in-memory application state.

A Match is a bounded contest: N hands (forced even so seat rotation balances),
chips as cumulative signed P&L from the human's perspective, winner decided by
chips — never by hands won. Hands won is displayed but decides nothing; the
whole lesson of Kuhn is that those two metrics diverge.

Live matches are held in memory (the current KuhnHistory node is process
state); completed hands and match totals are persisted through the Repository
so replays survive restarts. Mid-hand state is deliberately not persisted —
losing an unfinished hand to a restart is acceptable, losing history is not.
"""

from __future__ import annotations

import random
import time
import uuid

from pokerion.common.types import Action, InfoSetKey
from pokerion.game.base import History
from pokerion.game.kuhn import KuhnHistory

# Registry of available game variants
GAME_REGISTRY: dict[str, type[History]] = {
    "kuhn": KuhnHistory,
}

Strategy = dict[InfoSetKey, dict[Action, float]]

MATCH_TTL_SECONDS = 2 * 3600  # idle live matches get swept


class MatchError(ValueError):
    """A rule violation: illegal action, acting out of turn, match over."""


class Match:
    """A fixed-length contest between the human and the agent.

    Seats rotate every hand: the human is engine seat ``hand_index % 2``.
    Kuhn's first-acting seat loses 1/18 per hand at equilibrium, so an odd
    match would gift one side a structural edge — length is forced even.
    """

    MIN_LENGTH = 2
    MAX_LENGTH = 500
    DEFAULT_LENGTH = 50

    def __init__(
        self,
        variant: str,
        strategy: Strategy,
        length: int = DEFAULT_LENGTH,
        session_id: str | None = None,
    ):
        length = max(self.MIN_LENGTH, min(self.MAX_LENGTH, int(length)))
        if length % 2:
            length += 1

        # Full uuid4. Truncated to 8 hex chars this was 32 bits — ~50%
        # collision odds by 77k matches, and a collision cross-wires two
        # sessions: the loser's live match is evicted from the registry while
        # the winner's hands persist against the loser's DB row.
        self.id = str(uuid.uuid4())
        self.session_id = session_id
        self.variant = variant
        self.strategy = strategy  # frozen at creation; per-hand snapshot recorded anyway
        self.length = length
        self.history_cls = GAME_REGISTRY[variant]

        self.hand_index = 0        # completed hands
        self.human_chips = 0.0     # cumulative P&L, human perspective (zero-sum)
        self.hands: list[dict] = []
        self.last_activity = time.time()

        # Stored, not derived from hand_index: settling a hand increments the
        # index, but the seat must hold until the next deal — otherwise the
        # table re-orients while the finished hand is still on screen.
        self.human_seat = 0

        self._deal()

    # ------------------------------------------------------------ properties

    @property
    def agent_seat(self) -> int:
        return 1 - self.human_seat

    @property
    def is_complete(self) -> bool:
        return self.hand_index >= self.length

    @property
    def is_hand_terminal(self) -> bool:
        return self.current.is_terminal()

    @property
    def winner(self) -> str | None:
        if not self.is_complete:
            return None
        if self.human_chips > 0:
            return "human"
        if self.human_chips < 0:
            return "agent"
        return "draw"

    # ------------------------------------------------------------------ play
    def _deal(self) -> None:
        root = self.history_cls()
        self.current = root + root.sample_chance()
        self.states: list[dict] = [self.current.to_state_dict()]  # god view for replay
        # When seats rotate, the agent may hold the opening seat — it must act
        # before the human ever sees "your move".
        self._agent_autoplay()

    def _agent_autoplay(self) -> None:
        while not self.current.is_terminal() and (
            self.current.active_player() != self.human_seat
        ):
            self.current = self.current + self._agent_act()
            self.states.append(self.current.to_state_dict())

    def _agent_act(self) -> Action:
        key = self.current.info_set_key()
        if key in self.strategy:
            strat = self.strategy[key]
            actions = list(strat.keys())
            weights = list(strat.values())
            return random.choices(actions, weights=weights, k=1)[0]
        return random.choice(self.current.actions())

    def apply_action(self, action: Action) -> dict | None:
        """Apply a human action. Returns the completed hand record at terminal.

        Raises MatchError on any rule violation — the request layer turns that
        into a 4xx instead of silently corrupting the game tree.
        """
        self.last_activity = time.time()
        if self.is_complete:
            raise MatchError("match is complete")
        if self.current.is_terminal():
            raise MatchError("hand is over — deal the next one")
        if self.current.active_player() != self.human_seat:
            raise MatchError("not your turn")
        if action not in self.current.actions():
            raise MatchError(f"illegal action {action!r}")

        self.current = self.current + action
        self.states.append(self.current.to_state_dict())
        self._agent_autoplay()

        if self.current.is_terminal():
            return self._settle()
        return None

    def _settle(self) -> dict:
        delta = self.current.terminal_utility(self.human_seat)
        terminal = self.states[-1]
        record = {
            "hand_index": self.hand_index,
            "human_seat": self.human_seat,
            "chips_delta": delta,
            "winner_seat": terminal.get("winner"),
            "states": self.states,
            "strategy": self.strategy,
        }
        self.hands.append(record)
        self.human_chips += delta
        self.hand_index += 1  # also rotates human_seat for the next hand
        return record

    def new_hand(self) -> None:
        self.last_activity = time.time()
        if self.is_complete:
            raise MatchError("match is complete — start a new match")
        if not self.current.is_terminal():
            raise MatchError("current hand is still in progress")
        self.human_seat = self.hand_index % 2  # alternates 0,1,0,1,...
        self._deal()

    # ------------------------------------------------------------------ views
    def get_state(self) -> dict:
        """The human's view of the current hand plus match-level context."""
        state = self.current.to_state_dict(viewer=self.human_seat)
        state["match"] = self.meta()
        return state

    def meta(self) -> dict:
        return {
            "id": self.id,
            "variant": self.variant,
            "length": self.length,
            "hand_index": self.hand_index,
            "human_seat": self.human_seat,
            "human_chips": self.human_chips,
            "complete": self.is_complete,
            "winner": self.winner,
        }


class AppState:
    """Process-wide state: live matches in memory, everything durable in the repo."""

    def __init__(self):
        self.repo = None       # Repository, bound in app lifespan
        self.executor = None   # TrainingJobExecutor, bound in app lifespan
        self.matches: dict[str, Match] = {}

    def resolve_strategy(self, variant: str, session_id: str | None) -> Strategy:
        """Latest session-trained strategy, else the reference, else uniform.

        The uniform fallback only fires if the reference bake failed at boot —
        the game still works, the agent just plays randomly.
        """
        if self.repo is not None:
            row = self.repo.latest_strategy(variant, session_id)
            if row is not None:
                return row["payload"]
        return {}

    def create_match(self, variant: str, session_id: str, length: int) -> Match:
        self._sweep()
        strategy = self.resolve_strategy(variant, session_id)
        match = Match(
            variant=variant, strategy=strategy, length=length, session_id=session_id
        )
        # Persist BEFORE registering in memory. The old order evicted an
        # existing live match from the registry and only then discovered the
        # id was taken, leaving the victim's game unreachable mid-hand.
        if self.repo is not None:
            self.repo.create_match(match.id, session_id, variant, match.length)
        self.matches[match.id] = match
        return match

    def get_match(self, match_id: str, session_id: str | None = None) -> Match | None:
        match = self.matches.get(match_id)
        if match is None:
            return None
        # A match belongs to the session that created it.
        if session_id is not None and match.session_id != session_id:
            return None
        return match

    def persist_hand(self, match: Match, record: dict) -> None:
        if self.repo is None:
            return
        self.repo.record_hand(
            match_id=match.id,
            hand_index=record["hand_index"],
            human_seat=record["human_seat"],
            chips_delta=record["chips_delta"],
            winner_seat=record["winner_seat"],
            states=record["states"],
            strategy=record["strategy"],
        )
        self.repo.update_match(
            match.id, match.hand_index, match.human_chips, match.is_complete
        )

    def _sweep(self) -> None:
        cutoff = time.time() - MATCH_TTL_SECONDS
        stale = [mid for mid, m in self.matches.items() if m.last_activity < cutoff]
        for mid in stale:
            del self.matches[mid]


app_state = AppState()
