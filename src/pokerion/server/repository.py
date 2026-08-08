"""SQLite persistence for sessions, matches, replays, training jobs and strategies.

Design notes, in order of importance:

- `snapshots` deliberately has NO strategy column. Snapshots are metrics only
  (iteration, exploitability, game values). Retaining a full strategy dict per
  snapshot is what made session memory balloon and is unusable past Kuhn —
  the schema enforces what convention would eventually forget.

- `strategies.session_id IS NULL` marks the reference strategy baked at boot.
  A visitor's own training result has their session id. Play resolves "latest
  for this session, else reference" — one code path for both tiers.

- Everything goes through one connection guarded by a lock. SQLite in WAL mode
  handles our write rates trivially; the lock exists because the training
  executor writes from worker threads.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from pokerion.common.types import Action, InfoSetKey

Strategy = dict[InfoSetKey, dict[Action, float]]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    last_seen  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS matches (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL REFERENCES sessions(id),
    variant      TEXT NOT NULL,
    length       INTEGER NOT NULL,
    hand_index   INTEGER NOT NULL DEFAULT 0,
    human_chips  REAL NOT NULL DEFAULT 0,
    created_at   REAL NOT NULL,
    completed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_matches_session ON matches(session_id, created_at);

CREATE TABLE IF NOT EXISTS hands (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id    TEXT NOT NULL REFERENCES matches(id),
    hand_index  INTEGER NOT NULL,
    human_seat  INTEGER NOT NULL,
    chips_delta REAL NOT NULL,
    winner_seat INTEGER,
    states      TEXT NOT NULL,   -- JSON: god-mode state log for replay
    strategy    TEXT NOT NULL    -- JSON: agent strategy at the time of this hand
);
CREATE INDEX IF NOT EXISTS idx_hands_match ON hands(match_id, hand_index);

CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    variant         TEXT NOT NULL,
    status          TEXT NOT NULL,   -- queued | running | done | error
    iterations_req  INTEGER NOT NULL,
    iterations_done INTEGER NOT NULL DEFAULT 0,
    budget_seconds  REAL NOT NULL,
    error           TEXT,
    created_at      REAL NOT NULL,
    started_at      REAL,
    finished_at     REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_session ON jobs(session_id, status);

CREATE TABLE IF NOT EXISTS snapshots (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id         TEXT NOT NULL REFERENCES jobs(id),
    iteration      INTEGER NOT NULL,
    exploitability REAL NOT NULL,
    game_value_p0  REAL NOT NULL,
    game_value_p1  REAL NOT NULL,
    created_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_job ON snapshots(job_id, id);

CREATE TABLE IF NOT EXISTS strategies (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,                 -- NULL = the reference strategy
    job_id     TEXT,
    variant    TEXT NOT NULL,
    iterations INTEGER NOT NULL,
    payload    TEXT NOT NULL,        -- JSON strategy profile
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_strategies_lookup ON strategies(variant, session_id, id);
"""


def default_db_path() -> Path:
    return Path(os.environ.get("POKERION_DB", "data/pokerion.db"))


class Repository:
    def __init__(self, path: str | Path | None = None):
        db_path = Path(path) if path is not None else default_db_path()
        if str(db_path) != ":memory:":
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._closed = False
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            # NORMAL is the documented-safe pairing with WAL: durable against
            # process crash, only at risk from OS/power loss. FULL fsyncs on
            # every commit, and since the session middleware writes on most
            # requests that put a synchronous disk flush on the event loop.
            self._conn.execute("PRAGMA synchronous=NORMAL")
            # The REFERENCES clauses in the schema are inert without this —
            # orphan hands/snapshots would be accepted silently.
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._conn.close()

    class Closed(RuntimeError):
        """Raised instead of sqlite3.ProgrammingError when the repo is shut down.

        A worker thread can outlive close() by a hair. Surfacing a typed error
        lets callers distinguish "we are shutting down" from a real DB fault,
        rather than an opaque 'Cannot operate on a closed database' escaping a
        thread and printing a traceback at exit.
        """

    def _write(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            if self._closed:
                raise Repository.Closed("repository is closed")
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _read(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            if self._closed:
                raise Repository.Closed("repository is closed")
            return self._conn.execute(sql, params).fetchall()

    # ------------------------------------------------------------- sessions
    def touch_session(self, session_id: str) -> None:
        now = time.time()
        self._write(
            "INSERT INTO sessions (id, created_at, last_seen) VALUES (?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET last_seen = excluded.last_seen",
            (session_id, now, now),
        )

    # -------------------------------------------------------------- matches
    def create_match(self, match_id: str, session_id: str, variant: str, length: int) -> None:
        self._write(
            "INSERT INTO matches (id, session_id, variant, length, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (match_id, session_id, variant, length, time.time()),
        )

    def update_match(
        self, match_id: str, hand_index: int, human_chips: float, complete: bool
    ) -> None:
        self._write(
            "UPDATE matches SET hand_index = ?, human_chips = ?, completed_at = ? WHERE id = ?",
            (hand_index, human_chips, time.time() if complete else None, match_id),
        )

    def record_hand(
        self,
        match_id: str,
        hand_index: int,
        human_seat: int,
        chips_delta: float,
        winner_seat: int | None,
        states: list[dict],
        strategy: Strategy,
    ) -> None:
        self._write(
            "INSERT INTO hands (match_id, hand_index, human_seat, chips_delta, "
            "winner_seat, states, strategy) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                match_id,
                hand_index,
                human_seat,
                chips_delta,
                winner_seat,
                json.dumps(states),
                json.dumps(strategy),
            ),
        )

    # A session can hold up to MAX_MATCH_LENGTH hands per match across many
    # matches, and every hand carries a full state log plus a strategy copy.
    # Returning all of it unbounded meant a single request could serialize tens
    # of megabytes while holding the connection lock — measured at 0.49s of
    # fully blocking work and a 24 MB response for one 40-match session.
    MAX_MATCHES_RETURNED = 50

    def _hand_row(self, h: sqlite3.Row) -> dict:
        return {
            "hand_index": h["hand_index"],
            "human_seat": h["human_seat"],
            "chips_delta": h["chips_delta"],
            "winner_seat": h["winner_seat"],
            "states": json.loads(h["states"]),
            "strategy": json.loads(h["strategy"]),
        }

    def _match_row(self, m: sqlite3.Row, hands: list[dict]) -> dict:
        return {
            "id": m["id"],
            "variant": m["variant"],
            "length": m["length"],
            "hand_index": m["hand_index"],
            "human_chips": m["human_chips"],
            "complete": m["completed_at"] is not None,
            "hands": hands,
        }

    def get_match_with_hands(self, match_id: str, session_id: str) -> dict | None:
        """One match, fetched by id — not by scanning the session's history.

        The route used to call list_matches_with_hands() and linear-scan the
        result, so asking for one match cost the entire session.
        """
        rows = self._read(
            "SELECT * FROM matches WHERE id = ? AND session_id = ?",
            (match_id, session_id),
        )
        if not rows:
            return None
        hands = self._read(
            "SELECT * FROM hands WHERE match_id = ? ORDER BY hand_index", (match_id,)
        )
        return self._match_row(rows[0], [self._hand_row(h) for h in hands])

    def list_matches_with_hands(self, session_id: str) -> list[dict]:
        """Recent matches for this session, oldest first, hands inlined."""
        matches = self._read(
            "SELECT * FROM matches WHERE session_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (session_id, self.MAX_MATCHES_RETURNED),
        )
        out = []
        for m in reversed(matches):  # DESC+LIMIT keeps the newest; show oldest first
            hands = self._read(
                "SELECT * FROM hands WHERE match_id = ? ORDER BY hand_index",
                (m["id"],),
            )
            out.append(self._match_row(m, [self._hand_row(h) for h in hands]))
        return out

    # ----------------------------------------------------------------- jobs
    def create_job(
        self, session_id: str, variant: str, iterations: int, budget_seconds: float
    ) -> str:
        # Full uuid4, not a truncation. An 8-hex-char id is 32 bits: ~50%
        # collision odds by 77k rows, and a collision here would expose one
        # session's training snapshots to another.
        job_id = str(uuid.uuid4())
        self._write(
            "INSERT INTO jobs (id, session_id, variant, status, iterations_req, "
            "budget_seconds, created_at) VALUES (?, ?, ?, 'queued', ?, ?, ?)",
            (job_id, session_id, variant, iterations, budget_seconds, time.time()),
        )
        return job_id

    def job_started(self, job_id: str) -> None:
        self._write(
            "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
            (time.time(), job_id),
        )

    def job_finished(self, job_id: str, iterations_done: int, error: str | None = None) -> None:
        self._write(
            "UPDATE jobs SET status = ?, iterations_done = ?, error = ?, finished_at = ? "
            "WHERE id = ?",
            ("error" if error else "done", iterations_done, error, time.time(), job_id),
        )

    def get_job(self, job_id: str) -> dict | None:
        rows = self._read("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return dict(rows[0]) if rows else None

    def session_has_active_job(self, session_id: str, max_age_seconds: float = 120.0) -> bool:
        """True only for a job that is BOTH active and recent.

        The age guard is a safety net, not the primary fix: a job row stranded
        at 'running' by an unclean exit would otherwise gate this session out of
        training forever (the cookie lives 6 months). reconcile_orphaned_jobs()
        at startup is the real repair; this bounds the damage if anything ever
        strands a row while the process is up.
        """
        rows = self._read(
            "SELECT 1 FROM jobs WHERE session_id = ? AND status IN ('queued', 'running') "
            "AND created_at > ? LIMIT 1",
            (session_id, time.time() - max_age_seconds),
        )
        return bool(rows)

    def reconcile_orphaned_jobs(self) -> int:
        """Fail jobs left mid-flight by a previous process. Called at startup.

        A container restart (every CI deploy does one) kills running worker
        threads without unwinding them, so their rows keep status 'running'
        forever. Nothing else ever clears them.
        """
        cur = self._write(
            "UPDATE jobs SET status = 'error', error = 'interrupted by restart', "
            "finished_at = ? WHERE status IN ('queued', 'running')",
            (time.time(),),
        )
        return cur.rowcount

    # ------------------------------------------------------------ snapshots
    def add_snapshot(
        self, job_id: str, iteration: int, exploitability: float, game_values: list[float]
    ) -> None:
        self._write(
            "INSERT INTO snapshots (job_id, iteration, exploitability, game_value_p0, "
            "game_value_p1, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, iteration, exploitability, game_values[0], game_values[1], time.time()),
        )

    def snapshots_after(self, job_id: str, after_id: int = 0) -> list[dict]:
        rows = self._read(
            "SELECT * FROM snapshots WHERE job_id = ? AND id > ? ORDER BY id",
            (job_id, after_id),
        )
        return [dict(r) for r in rows]

    # ----------------------------------------------------------- strategies
    def save_strategy(
        self,
        variant: str,
        iterations: int,
        strategy: Strategy,
        session_id: str | None = None,
        job_id: str | None = None,
    ) -> None:
        self._write(
            "INSERT INTO strategies (session_id, job_id, variant, iterations, payload, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, job_id, variant, iterations, json.dumps(strategy), time.time()),
        )

    def latest_strategy(self, variant: str, session_id: str | None) -> dict | None:
        """The session's own latest result, falling back to the reference.

        Returns {"payload": Strategy, "iterations": int, "source": "session"|"reference"}.
        """
        if session_id is not None:
            rows = self._read(
                "SELECT payload, iterations FROM strategies "
                "WHERE variant = ? AND session_id = ? ORDER BY id DESC LIMIT 1",
                (variant, session_id),
            )
            if rows:
                return {
                    "payload": json.loads(rows[0]["payload"]),
                    "iterations": rows[0]["iterations"],
                    "source": "session",
                }
        rows = self._read(
            "SELECT payload, iterations FROM strategies "
            "WHERE variant = ? AND session_id IS NULL ORDER BY id DESC LIMIT 1",
            (variant,),
        )
        if rows:
            return {
                "payload": json.loads(rows[0]["payload"]),
                "iterations": rows[0]["iterations"],
                "source": "reference",
            }
        return None

    # ------------------------------------------------------------- retention
    def sweep(self, max_age_days: float = 30.0) -> dict[str, int]:
        """Delete data belonging to sessions that stopped visiting long ago.

        Nothing in this app previously deleted anything: sessions, matches,
        hands, jobs and snapshots all grew forever on a 20 GB root volume.
        Reference strategies (session_id IS NULL) are never swept.
        """
        cutoff = time.time() - max_age_days * 86400
        stale = [
            r["id"] for r in self._read(
                "SELECT id FROM sessions WHERE last_seen < ?", (cutoff,)
            )
        ]
        if not stale:
            return {"sessions": 0, "matches": 0, "hands": 0, "jobs": 0}

        deleted = {"sessions": 0, "matches": 0, "hands": 0, "jobs": 0}
        with self._lock:
            cur = self._conn.cursor()
            for i in range(0, len(stale), 500):  # keep the SQL variable count sane
                chunk = stale[i : i + 500]
                marks = ",".join("?" * len(chunk))
                # Children first: foreign_keys=ON is enabled, and there is no
                # ON DELETE CASCADE in the schema.
                cur.execute(
                    f"DELETE FROM hands WHERE match_id IN "
                    f"(SELECT id FROM matches WHERE session_id IN ({marks}))", chunk
                )
                deleted["hands"] += cur.rowcount
                cur.execute(
                    f"DELETE FROM snapshots WHERE job_id IN "
                    f"(SELECT id FROM jobs WHERE session_id IN ({marks}))", chunk
                )
                cur.execute(f"DELETE FROM matches WHERE session_id IN ({marks})", chunk)
                deleted["matches"] += cur.rowcount
                cur.execute(f"DELETE FROM jobs WHERE session_id IN ({marks})", chunk)
                deleted["jobs"] += cur.rowcount
                cur.execute(f"DELETE FROM strategies WHERE session_id IN ({marks})", chunk)
                cur.execute(f"DELETE FROM sessions WHERE id IN ({marks})", chunk)
                deleted["sessions"] += cur.rowcount
            self._conn.commit()
        return deleted

    def has_reference(self, variant: str) -> bool:
        return bool(
            self._read(
                "SELECT 1 FROM strategies WHERE variant = ? AND session_id IS NULL LIMIT 1",
                (variant,),
            )
        )
