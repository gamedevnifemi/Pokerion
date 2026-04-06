"""Vanilla CFR — full game tree traversal for Nash equilibrium computation.

Game-agnostic: works with any game that implements the History protocol.
Enumerates all chance outcomes (no sampling).
"""

from collections.abc import Callable

from pokerion.common.types import Action, InfoSetKey
from pokerion.game.base import History
from pokerion.solver.info_set import InfoSetNode


class VanillaCFR:
    """Counterfactual Regret Minimization solver.

    Traverses the complete game tree on each iteration, accumulating
    regrets at each information set. The average strategy converges
    to a Nash equilibrium.
    """

    def __init__(self, create_root: Callable[[], History], num_players: int = 2):
        self.create_root = create_root
        self.num_players = num_players
        self.info_sets: dict[InfoSetKey, InfoSetNode] = {}
        self.iterations = 0

    def train(self, iterations: int, callback: Callable[[int], None] | None = None):
        """Run `iterations` of CFR. Optionally call `callback(i)` each iteration."""
        for i in range(iterations):
            for player in range(self.num_players):
                reach = [1.0] * self.num_players
                self._cfr(self.create_root(), player, reach)
            self.iterations += 1
            if callback:
                callback(self.iterations)

    def _cfr(
        self,
        h: History,
        traverser: int,
        reach_probs: list[float],
    ) -> float:
        """Recursive CFR traversal. Returns counterfactual value for traverser."""
        if h.is_terminal():
            return h.terminal_utility(traverser)

        if h.is_chance():
            value = 0.0
            for action, prob in h.chance_actions():
                value += prob * self._cfr(h + action, traverser, reach_probs)
            return value

        player = h.active_player()
        actions = h.actions()
        key = h.info_set_key()

        # Get or create info set node
        if key not in self.info_sets:
            self.info_sets[key] = InfoSetNode(key, actions)
        node = self.info_sets[key]

        strategy = node.current_strategy()

        # Compute counterfactual value of each action
        action_values: dict[Action, float] = {}
        node_value = 0.0

        for a in actions:
            new_reach = reach_probs.copy()
            new_reach[player] *= strategy[a]
            action_values[a] = self._cfr(h + a, traverser, new_reach)
            node_value += strategy[a] * action_values[a]

        # Update regrets and strategy sum only for the traversing player
        if player == traverser:
            # Opponent reach probability (product of all non-traverser reaches)
            opponent_reach = 1.0
            for p in range(self.num_players):
                if p != traverser:
                    opponent_reach *= reach_probs[p]

            for a in actions:
                regret = action_values[a] - node_value
                node.regret_sum[a] += opponent_reach * regret

            node.accumulate_strategy(strategy, reach_probs[player])

        return node_value

    def get_strategy(self) -> dict[InfoSetKey, dict[Action, float]]:
        """Returns the average strategy profile (Nash equilibrium approximation)."""
        return {key: node.average_strategy() for key, node in self.info_sets.items()}

    def get_current_strategy(self) -> dict[InfoSetKey, dict[Action, float]]:
        """Returns the current regret-matched strategy (not the average)."""
        return {key: node.current_strategy() for key, node in self.info_sets.items()}

    def expected_utility(self) -> list[float]:
        """Compute game value for each player under the average strategy."""
        values = []
        for player in range(self.num_players):
            v = self._eval(self.create_root(), player)
            values.append(v)
        return values

    def _eval(self, h: History, player: int) -> float:
        """Evaluate expected utility under the average strategy."""
        if h.is_terminal():
            return h.terminal_utility(player)

        if h.is_chance():
            value = 0.0
            for action, prob in h.chance_actions():
                value += prob * self._eval(h + action, player)
            return value

        key = h.info_set_key()
        node = self.info_sets[key]
        strategy = node.average_strategy()

        value = 0.0
        for a in h.actions():
            value += strategy[a] * self._eval(h + a, player)
        return value

    def exploitability(self) -> float:
        """Compute exploitability: how much a best-response opponent can gain.

        Lower is better. 0 = perfect Nash equilibrium.
        Returns the sum of both players' best-response improvements.

        Uses info-set-level best response (not history-level), which is correct
        for imperfect information games. The BR player must pick one action per
        information set — they can't distinguish histories within the same set.
        """
        total = 0.0
        for player in range(self.num_players):
            br_strategy = self._compute_br_strategy(player)
            br_value = self._eval_with_pure(self.create_root(), player, br_strategy)
            eq_value = self._eval(self.create_root(), player)
            total += br_value - eq_value
        return total

    def _compute_br_strategy(self, br_player: int) -> dict[InfoSetKey, Action]:
        """Compute best response pure strategy for br_player, bottom-up.

        Process info sets from deepest to shallowest. For each info set,
        evaluate each action's expected value (using already-determined BR
        actions at deeper info sets, opponent's average strategy elsewhere),
        then pick the best action.
        """
        # Step 1: Find all BR player info sets and their depths
        info_depths: dict[InfoSetKey, int] = {}
        self._find_info_set_depths(self.create_root(), br_player, 0, info_depths)

        # Step 2: Process deepest first
        br_actions: dict[InfoSetKey, Action] = {}
        for key in sorted(info_depths, key=info_depths.get, reverse=True):
            node = self.info_sets[key]
            best_action = node.actions[0]
            best_value = float("-inf")
            for a in node.actions:
                # Evaluate game value if BR player plays `a` at this info set,
                # already-determined BR at deeper info sets, avg strategy elsewhere
                extended = {**br_actions, key: a}
                v = self._eval_with_pure(self.create_root(), br_player, extended)
                if v > best_value:
                    best_value = v
                    best_action = a
            br_actions[key] = best_action

        return br_actions

    def _find_info_set_depths(
        self, h: History, br_player: int, depth: int,
        result: dict[InfoSetKey, int],
    ):
        """Collect all info sets belonging to br_player with their tree depth."""
        if h.is_terminal():
            return
        if h.is_chance():
            for a, _ in h.chance_actions():
                self._find_info_set_depths(h + a, br_player, depth, result)
            return

        if h.active_player() == br_player:
            key = h.info_set_key()
            if key not in result:
                result[key] = depth

        for a in h.actions():
            self._find_info_set_depths(h + a, br_player, depth + 1, result)

    def _eval_with_pure(
        self, h: History, player: int,
        pure_actions: dict[InfoSetKey, Action],
    ) -> float:
        """Evaluate game value when `player` plays pure strategy at specified
        info sets and average strategy elsewhere. Opponents play average strategy."""
        if h.is_terminal():
            return h.terminal_utility(player)

        if h.is_chance():
            return sum(
                p * self._eval_with_pure(h + a, player, pure_actions)
                for a, p in h.chance_actions()
            )

        acting = h.active_player()
        key = h.info_set_key()
        node = self.info_sets[key]

        if acting == player and key in pure_actions:
            return self._eval_with_pure(h + pure_actions[key], player, pure_actions)

        strategy = node.average_strategy()
        return sum(
            strategy[a] * self._eval_with_pure(h + a, player, pure_actions)
            for a in h.actions()
        )
