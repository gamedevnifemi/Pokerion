"""Training API routes — thin faces over the job executor.

The WebSocket is a *reader*: it submits a job and then tails the snapshot
table. The run belongs to the executor, so a dropped socket no longer kills
the training it started, and reconnecting mid-run just resumes the tail.
"""

import asyncio

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from pokerion.server.jobs import JobRejected
from pokerion.server.session import (
    SESSION_COOKIE,
    client_key,
    is_valid_session_id,
    new_session_id,
)
from pokerion.server.state import GAME_REGISTRY, app_state

router = APIRouter(prefix="/api")

_POLL_INTERVAL = 0.15   # seconds between snapshot-table polls while streaming
_REST_TIMEOUT = 60.0    # cap on the synchronous /api/train wait

# A socket costs almost nothing to open but drives a poll loop that hits SQLite
# twice per interval, so connections are the resource to bound here.
MAX_WS_CONNECTIONS = 24
_ws_connections = 0


def _snapshot_payload(row: dict) -> dict:
    return {
        "iteration": row["iteration"],
        "exploitability": row["exploitability"],
        "game_values": [row["game_value_p0"], row["game_value_p1"]],
    }


@router.post("/train")
async def start_training(request: Request, iterations: int = 1000, variant: str = "kuhn"):
    """Synchronous fallback: run a job to completion, return all snapshots."""
    try:
        job_id, future = app_state.executor.submit(
            request.state.session_id,
            variant,
            iterations,
            client_key=request.state.client_key,
        )
    except JobRejected as exc:
        return JSONResponse({"error": str(exc)}, status_code=429)

    try:
        # Bounded: without a timeout a queued job parks the request (and a
        # worker connection) for as long as the backlog takes to drain.
        await asyncio.wait_for(asyncio.wrap_future(future), timeout=_REST_TIMEOUT)
    except (TimeoutError, asyncio.CancelledError):
        return JSONResponse(
            {"error": "training did not finish in time", "job_id": job_id},
            status_code=504,
        )

    job = app_state.repo.get_job(job_id)
    if job is None:
        return JSONResponse({"error": "job disappeared"}, status_code=500)
    if job["status"] == "error":
        # Previously this returned 200 with the *previous* strategy, making a
        # failed run indistinguishable from a successful one.
        return JSONResponse(
            {"error": job["error"] or "training failed", "job_id": job_id},
            status_code=500,
        )

    snapshots = app_state.repo.snapshots_after(job_id)
    strategy_row = app_state.repo.latest_strategy(variant, request.state.session_id)
    return {
        "job_id": job_id,
        "total_iterations": job["iterations_done"],
        "snapshots": [_snapshot_payload(s) for s in snapshots],
        "strategy": strategy_row["payload"] if strategy_row else {},
    }


@router.get("/strategy")
async def get_strategy(request: Request, variant: str = "kuhn"):
    """The strategy Play would use right now: session-trained, else reference."""
    row = app_state.repo.latest_strategy(variant, request.state.session_id)
    if row is None:
        return {"iterations": 0, "source": "none", "strategy": {}}
    return {
        "iterations": row["iterations"],
        "source": row["source"],
        "strategy": row["payload"],
    }


@router.get("/games")
async def list_games():
    return {"variants": list(GAME_REGISTRY.keys())}


@router.websocket("/ws/train")
async def ws_train(ws: WebSocket):
    """Submit a job, then stream its snapshots as they land in the store."""
    global _ws_connections

    if _ws_connections >= MAX_WS_CONNECTIONS:
        await ws.close(code=1013)  # try again later
        return

    await ws.accept()
    _ws_connections += 1
    try:
        # HTTP middleware does not run for WebSockets — read the cookie here,
        # with the same validation, so a crafted value cannot become a row key.
        raw = ws.cookies.get(SESSION_COOKIE)
        session_id = raw if is_valid_session_id(raw) else new_session_id()
        key = client_key(ws)
        app_state.repo.touch_session(session_id)

        while True:
            req = await ws.receive_json()
            if not isinstance(req, dict):
                await ws.send_json({"type": "error", "message": "expected an object"})
                continue

            iterations = req.get("iterations", 1000)
            variant = req.get("variant", "kuhn")
            if not isinstance(variant, str):
                await ws.send_json({"type": "error", "message": "variant must be a string"})
                continue

            try:
                job_id, _future = app_state.executor.submit(
                    session_id, variant, iterations, client_key=key
                )
            except JobRejected as exc:
                await ws.send_json({"type": "error", "message": str(exc)})
                continue

            await ws.send_json({"type": "started", "job_id": job_id})

            last_snapshot_id = 0
            while True:
                rows = app_state.repo.snapshots_after(job_id, last_snapshot_id)
                for row in rows:
                    last_snapshot_id = row["id"]
                    payload = {"type": "progress", **_snapshot_payload(row)}
                    # Live strategy comes from the in-process runner for the
                    # charts; it is never persisted with the snapshot.
                    live = app_state.executor.live_strategy(job_id)
                    if live is not None:
                        payload["strategy"] = live
                    await ws.send_json(payload)

                job = app_state.repo.get_job(job_id)
                if job is None or job["status"] in ("done", "error"):
                    if job is None or job["status"] == "error":
                        message = (job or {}).get("error") or "training failed"
                        await ws.send_json({"type": "error", "message": message})
                        break
                    strategy_row = app_state.repo.latest_strategy(variant, session_id)
                    await ws.send_json(
                        {
                            "type": "done",
                            "iteration": job["iterations_done"],
                            "strategy": strategy_row["payload"] if strategy_row else {},
                        }
                    )
                    break

                await asyncio.sleep(_POLL_INTERVAL)

    except WebSocketDisconnect:
        pass  # the job keeps running; the store keeps the snapshots
    except Exception:
        # A malformed frame must close this socket, not raise past the handler.
        try:
            await ws.close(code=1011)
        except Exception:
            pass
    finally:
        _ws_connections -= 1
