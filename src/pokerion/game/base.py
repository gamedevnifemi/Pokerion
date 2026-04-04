from abc import ABC, abstractmethod

from pokerion.common.types import Action, InfoSetKey, Player


class History(ABC):
    """A node in the game tree, represented as the sequence of actions from root.

    Immutable: h + action creates a new child node without modifying h.
    The CFR solver traverses the tree using this interface only.
    """

    @abstractmethod
    def is_terminal(self) -> bool:
        """True if this node is a leaf (game is over)."""

    @abstractmethod
    def terminal_utility(self, player: Player) -> float:
        """Payoff for `player` at a terminal node. Only valid when is_terminal()."""

    @abstractmethod
    def active_player(self) -> Player:
        """Which player acts at this node (0-indexed). Not valid at terminal/chance nodes."""

    @abstractmethod
    def is_chance(self) -> bool:
        """True if this is a chance node (nature deals cards)."""

    @abstractmethod
    def chance_actions(self) -> list[tuple[Action, float]]:
        """All chance outcomes with their probabilities. Only valid at chance nodes.

        Used by vanilla CFR which enumerates all outcomes.
        """

    @abstractmethod
    def sample_chance(self) -> Action:
        """Sample one chance outcome. Used by MCCFR variants."""

    @abstractmethod
    def actions(self) -> list[Action]:
        """Legal actions for the active player at this decision node."""

    @abstractmethod
    def info_set_key(self) -> InfoSetKey:
        """String identifying what the active player can observe.

        Hides opponent private information. Two histories with the same
        info_set_key are indistinguishable to the active player.
        """

    @abstractmethod
    def __add__(self, action: Action) -> History:
        """Create a child node by appending an action. Does not modify self."""

    @abstractmethod
    def to_state_dict(self, viewer: Player | None = None) -> dict:
        """Serialize to a JSON-compatible dict for the visualizer.

        If viewer is set, hide information that player cannot see.
        If viewer is None, show everything (god mode / replay).
        """

    @property
    @abstractmethod
    def variant(self) -> str:
        """Game variant identifier, e.g. 'kuhn', 'leduc', 'holdem'."""

    @property
    @abstractmethod
    def num_players(self) -> int:
        """Number of players in this game."""
