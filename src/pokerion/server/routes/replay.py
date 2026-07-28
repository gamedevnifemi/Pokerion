"""Replay API routes — step through completed hands."""

from fastapi import APIRouter

from pokerion.server.state import app_state

router = APIRouter(prefix="/api/replay")


@router.get("/sessions")
async def list_sessions():
    """Every session played this server-run, each with its hands (god mode).

    Declared before /{game_id} so the literal path wins over the path param.
    Lets the frontend rebuild its replay history after a page reload.
    """
    return {
        "sessions": [
            {
                "id": session.id,
                "variant": session.variant,
                "hands": session.get_all_hands(),
                "strategy": session.strategy,
            }
            for session in app_state.games.values()
        ]
    }


@router.get("/{game_id}")
async def get_replay(game_id: str):
    """Get the full session history for step-through replay (god mode — shows all cards).

    `hands` is one state log per hand played, in the order they were dealt.
    """
    session = app_state.games.get(game_id)
    if not session:
        return {"error": "Game not found"}

    return {
        "game_id": game_id,
        "variant": session.variant,
        "hands": session.get_all_hands(),
        "strategy": session.strategy,
    }
