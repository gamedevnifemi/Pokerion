"""Replay API — session-scoped match history, read from the store.

Backed by SQLite rather than process memory, so replays survive restarts and
each visitor only ever sees their own matches.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from pokerion.server.state import app_state

router = APIRouter(prefix="/api/replay")


@router.get("/sessions")
async def list_sessions(request: Request):
    """Every match this visitor has played, each with its hands (god mode).

    Declared before /{match_id} so the literal path wins over the path param.
    """
    matches = app_state.repo.list_matches_with_hands(request.state.session_id)
    return {"sessions": matches}


@router.get("/{match_id}")
async def get_replay(request: Request, match_id: str):
    match = app_state.repo.get_match_with_hands(match_id, request.state.session_id)
    if match is None:
        return JSONResponse({"error": "match not found"}, status_code=404)
    return match
