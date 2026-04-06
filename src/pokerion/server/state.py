"""In-memory state for active training sessions and game sessions."""

import uuid

from pokerion.common.types import Action, InfoSetKey
from pokerion.game.base import History
from pokerion.game.kuhn import KuhnHistory
from pokerion.training.runner import TrainingRunner


# Registry of available game variants
GAME_REGISTRY: dict[str, type[History]] = {
    "kuhn": KuhnHistory,
}


class GameSession:
    """A single game being played between human and agent."""

    def __init__(self, variant: str, strategy: dict[InfoSetKey, dict[Action, float]]):
        self.id = str(uuid.uuid4())[:8]
        self.variant = variant
        self.strategy = strategy
        self.history_cls = GAME_REGISTRY[variant]
        self.states: list[dict] = []  # full replay log
        self._start_new_hand()

    def _start_new_hand(self):
        root = self.history_cls()
        # Deal cards (sample chance)
        deal = root.sample_chance()
        self.current = root + deal
        self._log_state()

    def _log_state(self):
        self.states.append(self.current.to_state_dict())

    def get_state(self, viewer: int | None = None) -> dict:
        return self.current.to_state_dict(viewer=viewer)

    def apply_action(self, action: Action) -> dict:
        """Apply human action, then let agent respond if it's agent's turn."""
        self.current = self.current + action
        self._log_state()

        # Agent responds until it's human's turn or game ends
        while not self.current.is_terminal() and not self.current.is_chance():
            if self.current.active_player() != self.human_player:
                agent_action = self._agent_act()
                self.current = self.current + agent_action
                self._log_state()
            else:
                break

        return self.get_state(viewer=self.human_player)

    def _agent_act(self) -> Action:
        """Agent picks action according to trained strategy (mixed)."""
        import random

        key = self.current.info_set_key()
        if key in self.strategy:
            strat = self.strategy[key]
            actions = list(strat.keys())
            weights = list(strat.values())
            return random.choices(actions, weights=weights, k=1)[0]
        # Fallback: uniform random
        actions = self.current.actions()
        return random.choice(actions)

    @property
    def human_player(self) -> int:
        return 0

    @property
    def is_terminal(self) -> bool:
        return self.current.is_terminal()

    def new_hand(self) -> dict:
        """Start a fresh hand, keep same session."""
        self._start_new_hand()
        return self.get_state(viewer=self.human_player)

    def get_replay(self) -> list[dict]:
        """Return all states with full information (god mode)."""
        return self.states


class AppState:
    """Global application state."""

    def __init__(self):
        self.trainer: TrainingRunner | None = None
        self.games: dict[str, GameSession] = {}
        self.training_variant: str = "kuhn"

    def get_or_create_trainer(self, variant: str = "kuhn") -> TrainingRunner:
        if self.trainer is None or self.training_variant != variant:
            game_cls = GAME_REGISTRY[variant]
            self.trainer = TrainingRunner(game_factory=game_cls, num_players=2)
            self.training_variant = variant
        return self.trainer

    def create_game(self, variant: str = "kuhn") -> GameSession:
        trainer = self.get_or_create_trainer(variant)
        strategy = trainer.get_strategy()
        session = GameSession(variant=variant, strategy=strategy)
        self.games[session.id] = session
        return session


app_state = AppState()
