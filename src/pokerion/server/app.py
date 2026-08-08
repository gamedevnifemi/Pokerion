"""FastAPI application — serves API and static frontend.

Lifespan owns the durable pieces: open the repository, start the job executor,
and bake the reference strategy if the store doesn't have one yet. The bake is
what makes Play work for a visitor who never clicks Train — and it happens at
boot rather than image build because the database is a volume the build can't
reach. It runs once per database, not once per boot.
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pokerion.server.jobs import TrainingJobExecutor
from pokerion.server.repository import Repository
from pokerion.server.routes import match, replay, training
from pokerion.server.session import SessionMiddleware
from pokerion.server.state import GAME_REGISTRY, app_state
from pokerion.training.runner import TrainingRunner

REFERENCE_ITERATIONS = 25_000  # ~5s of vanilla CFR on Kuhn; runs once per database


def _ensure_reference_strategies(repo: Repository) -> None:
    for variant, game_cls in GAME_REGISTRY.items():
        if repo.has_reference(variant):
            continue
        t0 = time.perf_counter()
        runner = TrainingRunner(game_factory=game_cls, num_players=2)
        runner.solver.train(REFERENCE_ITERATIONS)
        repo.save_strategy(
            variant=variant,
            iterations=REFERENCE_ITERATIONS,
            strategy=runner.get_strategy(),
            session_id=None,  # NULL session marks the reference
        )
        print(
            f"[pokerion] baked reference strategy for {variant!r}: "
            f"{REFERENCE_ITERATIONS} iterations in {time.perf_counter() - t0:.1f}s"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    repo = Repository()

    # Any job row still 'queued'/'running' belongs to a process that is gone.
    # Without this, session_has_active_job stays true for those visitors and
    # they can never train again — and every CI deploy creates a fresh batch.
    orphaned = repo.reconcile_orphaned_jobs()
    if orphaned:
        print(f"[pokerion] failed {orphaned} job(s) orphaned by the previous process")

    swept = repo.sweep()
    if any(swept.values()):
        print(f"[pokerion] retention sweep removed {swept}")

    _ensure_reference_strategies(repo)
    app_state.repo = repo
    app_state.executor = TrainingJobExecutor(repo)
    yield
    # Order matters: drain workers BEFORE closing the connection they write to.
    app_state.executor.shutdown()
    repo.close()


app = FastAPI(title="Pokerion", version="0.2.0", lifespan=lifespan)

app.add_middleware(SessionMiddleware, repo_getter=lambda: app_state.repo)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_frontend(request, call_next):
    """Stop the browser serving stale JS/CSS.

    StaticFiles only sends etag/last-modified, which lets Chrome reuse a
    memory-cached copy without revalidating — so frontend edits appear to do
    nothing until a hard refresh. The site is tiny; correctness beats caching.
    """
    response = await call_next(request)
    if not request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


@app.get("/api/version")
async def version():
    """The commit this image was built from. CI's deploy verification polls
    this until production reports the sha it just pushed."""
    return {"sha": os.environ.get("POKERION_GIT_SHA", "dev")}


# API routes
app.include_router(training.router)
app.include_router(match.router)
app.include_router(replay.router)

# Serve frontend static files. The env override exists because an installed
# package's __file__ lives in site-packages, nowhere near the repo layout.
_frontend_env = os.environ.get("POKERION_FRONTEND")
frontend_dir = (
    Path(_frontend_env)
    if _frontend_env
    else Path(__file__).resolve().parent.parent.parent.parent / "frontend"
)
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
