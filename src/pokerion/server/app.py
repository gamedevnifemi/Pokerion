"""FastAPI application — serves API and static frontend."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pokerion.server.routes import game, replay, training

app = FastAPI(title="Pokerion", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(training.router)
app.include_router(game.router)
app.include_router(replay.router)

# Serve frontend static files
frontend_dir = Path(__file__).resolve().parent.parent.parent.parent / "frontend"
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
