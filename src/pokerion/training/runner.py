"""Training runner — orchestrates solver + game with progress tracking."""

from collections.abc import Callable

from pokerion.game.base import History
from pokerion.solver.vanilla_cfr import VanillaCFR


class TrainingRunner:
    """Manages a CFR training session with progress callbacks and snapshots."""

    def __init__(
        self,
        game_factory: Callable[[], History],
        num_players: int = 2,
    ):
        self.solver = VanillaCFR(create_root=game_factory, num_players=num_players)
        self.history: list[dict] = []  # per-snapshot metrics

    def train(
        self,
        iterations: int,
        snapshot_interval: int = 100,
        on_snapshot: Callable[[dict], None] | None = None,
    ):
        """Train for `iterations`, taking snapshots at regular intervals.

        on_snapshot receives a dict with:
            iteration, exploitability, game_values, strategy
        """
        target = self.solver.iterations + iterations

        def on_iter(i: int):
            if i % snapshot_interval == 0 or i == target:
                snap = self._take_snapshot()
                self.history.append(snap)
                if on_snapshot:
                    on_snapshot(snap)

        self.solver.train(iterations, callback=on_iter)

    def _take_snapshot(self) -> dict:
        game_values = self.solver.expected_utility()
        return {
            "iteration": self.solver.iterations,
            "exploitability": self.solver.exploitability(),
            "game_values": game_values,
            "strategy": self.solver.get_strategy(),
        }

    def get_strategy(self) -> dict:
        return self.solver.get_strategy()

    def get_current_strategy(self) -> dict:
        return self.solver.get_current_strategy()

    @property
    def iterations(self) -> int:
        return self.solver.iterations
