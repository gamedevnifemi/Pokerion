"""Match play API — human vs agent, bounded contests, chips as the verdict."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from pokerion.server.state import Match, MatchError, app_state

router = APIRouter(prefix="/api/match")


def _error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


@router.post("/new")
async def new_match(request: Request, variant: str = "kuhn", length: int = Match.DEFAULT_LENGTH):
    """Start a match. Length is clamped and forced even (seat-rotation fairness)."""
    session_id = request.state.session_id
    try:
        match = app_state.create_match(variant, session_id, length)
    except KeyError:
        return _error(f"unknown variant {variant!r}", 404)
    return {"match_id": match.id, "state": match.get_state()}


@router.post("/{match_id}/action")
async def take_action(request: Request, match_id: str, action: str):
    match = app_state.get_match(match_id, request.state.session_id)
    if match is None:
        return _error("match not found", 404)

    try:
        record = match.apply_action(action)
    except MatchError as exc:
        return _error(str(exc))

    result: dict = {"state": match.get_state()}
    if record is not None:
        app_state.persist_hand(match, record)
        result["terminal"] = True
        result["replay"] = record["states"]
        result["agent_strategy"] = record["strategy"]
        result["hand_meta"] = {
            "hand_index": record["hand_index"],
            "human_seat": record["human_seat"],
            "chips_delta": record["chips_delta"],
            "winner_seat": record["winner_seat"],
        }
    return result


@router.post("/{match_id}/new-hand")
async def new_hand(request: Request, match_id: str):
    match = app_state.get_match(match_id, request.state.session_id)
    if match is None:
        return _error("match not found", 404)
    try:
        match.new_hand()
    except MatchError as exc:
        return _error(str(exc))
    return {"state": match.get_state()}


@router.get("/{match_id}/state")
async def get_state(request: Request, match_id: str):
    match = app_state.get_match(match_id, request.state.session_id)
    if match is None:
        return _error("match not found", 404)
    return {"state": match.get_state()}
