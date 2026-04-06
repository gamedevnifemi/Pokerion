"""Replay API routes — step through completed hands."""

from fastapi import APIRouter

from pokerion.server.state import app_state

router = APIRouter(prefix="/api/replay")


@router.get("/{game_id}")
async def get_replay(game_id: str):
    """Get full hand history for step-through replay (god mode — shows all cards)."""
    session = app_state.games.get(game_id)
    if not session:
        return {"error": "Game not found"}

    return {
        "game_id": game_id,
        "variant": session.variant,
        "states": session.get_replay(),
        "strategy": session.strategy,
    }
