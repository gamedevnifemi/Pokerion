"""Information set node — tracks regret and strategy for one information set."""

from pokerion.common.types import Action, InfoSetKey


class InfoSetNode:
    """Stores cumulative regret and strategy sums for a single information set.

    The regret-matching algorithm converts cumulative regrets into a current
    strategy (behavior policy). The average strategy over all iterations
    converges to Nash equilibrium.
    """

    def __init__(self, key: InfoSetKey, actions: list[Action]):
        self.key = key
        self.actions = actions
        self.regret_sum: dict[Action, float] = {a: 0.0 for a in actions}
        self.strategy_sum: dict[Action, float] = {a: 0.0 for a in actions}

    def current_strategy(self) -> dict[Action, float]:
        """Regret matching: normalize positive regrets into a probability distribution.

        If all regrets are non-positive, return uniform distribution.
        """
        positive_regret = {a: max(0.0, self.regret_sum[a]) for a in self.actions}
        total = sum(positive_regret.values())

        if total > 0:
            return {a: positive_regret[a] / total for a in self.actions}
        # Uniform fallback
        n = len(self.actions)
        return {a: 1.0 / n for a in self.actions}

    def average_strategy(self) -> dict[Action, float]:
        """The time-averaged strategy — converges to Nash equilibrium."""
        total = sum(self.strategy_sum.values())

        if total > 0:
            return {a: self.strategy_sum[a] / total for a in self.actions}
        n = len(self.actions)
        return {a: 1.0 / n for a in self.actions}

    def accumulate_strategy(self, strategy: dict[Action, float], reach_prob: float):
        """Add current strategy weighted by the player's reach probability."""
        for a in self.actions:
            self.strategy_sum[a] += reach_prob * strategy[a]
