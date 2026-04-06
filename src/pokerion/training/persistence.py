"""Save and load trained strategy profiles."""

import json
from pathlib import Path

from pokerion.common.types import Action, InfoSetKey


def save_strategy(
    strategy: dict[InfoSetKey, dict[Action, float]],
    path: str | Path,
    metadata: dict | None = None,
):
    """Save a strategy profile to JSON."""
    data = {
        "metadata": metadata or {},
        "strategy": strategy,
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_strategy(path: str | Path) -> dict[InfoSetKey, dict[Action, float]]:
    """Load a strategy profile from JSON."""
    data = json.loads(Path(path).read_text())
    return data["strategy"]
