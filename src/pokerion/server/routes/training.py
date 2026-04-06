"""Training API routes."""

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from pokerion.server.state import app_state

router = APIRouter(prefix="/api")


@router.post("/train")
async def start_training(iterations: int = 1000, variant: str = "kuhn"):
    """Train the CFR solver for the specified number of iterations."""
    trainer = app_state.get_or_create_trainer(variant)

    snapshots = []

    def on_snapshot(snap):
        snapshots.append({
            "iteration": snap["iteration"],
            "exploitability": snap["exploitability"],
            "game_values": snap["game_values"],
        })

    # Run training in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: trainer.train(iterations, snapshot_interval=max(1, iterations // 20), on_snapshot=on_snapshot),
    )

    return {
        "total_iterations": trainer.iterations,
        "snapshots": snapshots,
        "strategy": trainer.get_strategy(),
    }


@router.get("/strategy")
async def get_strategy(variant: str = "kuhn"):
    """Get the current trained strategy profile."""
    trainer = app_state.get_or_create_trainer(variant)
    return {
        "iterations": trainer.iterations,
        "strategy": trainer.get_strategy(),
    }


@router.get("/games")
async def list_games():
    """List available game variants."""
    from pokerion.server.state import GAME_REGISTRY

    return {"variants": list(GAME_REGISTRY.keys())}


@router.websocket("/ws/train")
async def ws_train(ws: WebSocket):
    """WebSocket endpoint for live training with real-time strategy updates."""
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            iterations = data.get("iterations", 1000)
            variant = data.get("variant", "kuhn")
            batch_size = data.get("batch_size", 50)

            trainer = app_state.get_or_create_trainer(variant)

            remaining = iterations
            while remaining > 0:
                batch = min(batch_size, remaining)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda b=batch: trainer.train(b, snapshot_interval=b),
                )
                remaining -= batch

                snap = {
                    "type": "progress",
                    "iteration": trainer.iterations,
                    "exploitability": trainer.solver.exploitability(),
                    "game_values": trainer.solver.expected_utility(),
                    "strategy": trainer.get_strategy(),
                }
                await ws.send_json(snap)

            await ws.send_json({"type": "done", "iteration": trainer.iterations})

    except WebSocketDisconnect:
        pass
