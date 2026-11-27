"""Tables of bots, scored exactly.

- `table`: fixed strategies, scored over every seating.
- `match`: many hands between agents that may adapt, scored hand by hand.
- `agents`, `memory`, `exploitation`: an adapting agent and its two swappable parts.
- `bots`: simple fixed strategies to play against.
"""

from pokerion.arena.agents import AdaptiveAgent
from pokerion.arena.exploitation import ExploitationRule, MixWithBestResponse, PlayTheCore
from pokerion.arena.match import (
    Agent,
    Decision,
    HandView,
    MatchResult,
    PlayedHand,
    play_match,
    play_matches,
)
from pokerion.arena.memory import CountingMemory, OpponentMemory
from pokerion.arena.table import (
    Entrant,
    TableResult,
    best_reply_to_table,
    evaluate_table,
    table_mix_curve,
)

__all__ = [
    "AdaptiveAgent",
    "Agent",
    "CountingMemory",
    "Decision",
    "Entrant",
    "ExploitationRule",
    "HandView",
    "MatchResult",
    "MixWithBestResponse",
    "OpponentMemory",
    "PlayTheCore",
    "PlayedHand",
    "TableResult",
    "best_reply_to_table",
    "evaluate_table",
    "play_match",
    "play_matches",
    "table_mix_curve",
]
