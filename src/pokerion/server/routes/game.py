"""Game play API routes — human vs agent."""

from fastapi import APIRouter

from pokerion.server.state import app_state

router = APIRouter(prefix="/api/game")


@router.post("/new")
async def new_game(variant: str = "kuhn"):
    """Create a new game session. Returns the initial state (human sees own card)."""
    session = app_state.create_game(variant)
    return {
        "game_id": session.id,
        "state": session.get_state(viewer=session.human_player),
    }


@router.post("/{game_id}/action")
async def take_action(game_id: str, action: str):
    """Submit a player action. Agent responds automatically."""
    session = app_state.games.get(game_id)
    if not session:
        return {"error": "Game not found"}

    state = session.apply_action(action)

    result = {"state": state}
    if session.is_terminal:
        result["terminal"] = True
        result["replay"] = session.get_replay()
        # Include agent strategy at each decision point for replay
        result["agent_strategy"] = session.strategy

    return result


@router.post("/{game_id}/new-hand")
async def new_hand(game_id: str):
    """Deal a new hand in the same session."""
    session = app_state.games.get(game_id)
    if not session:
        return {"error": "Game not found"}

    state = session.new_hand()
    return {"state": state}


@router.get("/{game_id}/state")
async def get_state(game_id: str):
    """Get current game state."""
    session = app_state.games.get(game_id)
    if not session:
        return {"error": "Game not found"}

    return {"state": session.get_state(viewer=session.human_player)}
