"""Training as jobs, not requests.

The request that starts training returns a job id immediately; the run itself
happens on a worker thread, writes metric snapshots to the repository, and
persists the final strategy when it finishes. WebSocket clients *read* progress
from the store — they never own the run, so a dropped connection no longer
kills the training it started.

Two caps bound the DoS surface:

- ``budget_seconds`` is the real limit. Wall-clock means the same thing in
  every game — "5,000 iterations" is a Kuhn-shaped unit that becomes a
  different promise the day Leduc lands.
- ``max_iterations`` is belt-and-braces on top.

Snapshot cadence is time-based, not batch-based. The old per-batch cadence ran
a full best-response computation ~50 times per run; at Kuhn scale that is 19%
of wall clock, at Leduc scale the measurement would dominate the training.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

from pokerion.server.repository import Repository
from pokerion.server.state import GAME_REGISTRY
from pokerion.training.runner import TrainingRunner


class JobRejected(Exception):
    """Submission refused (unknown variant, already training, system busy).

    Deliberately NOT a ValueError: it used to be, which meant a plain
    ValueError from int('abc') on unvalidated WebSocket input was caught by
    `except JobRejected` handlers that had no business catching it.
    """


class TrainingJobExecutor:
    MAX_ITERATIONS = 100_000
    MAX_BUDGET_SECONDS = 15.0
    DEFAULT_BUDGET_SECONDS = 10.0
    TRAIN_BATCH = 200                 # internal step between time checks
    SNAPSHOT_MIN_INTERVAL = 0.25      # seconds between metric snapshots

    # Admission control. Two workers means the queue is the real resource:
    # 8 pending jobs is ~1 minute of backlog, after which callers get a 429
    # instead of parking on a future that will not run for hours.
    MAX_JOBS_PER_CLIENT = 2
    MAX_QUEUED_TOTAL = 8

    # Concurrency caps bound simultaneous work, but /api/train blocks until its
    # job finishes — so a serial loop releases its slot each time and can pin
    # the CPU indefinitely, one job after another. This budgets total compute
    # per client over a rolling window: 6 jobs/minute at a 15s ceiling is at
    # most ~90 CPU-seconds/minute from any one source.
    RATE_WINDOW_SECONDS = 60.0
    MAX_JOBS_PER_WINDOW = 6

    def __init__(self, repo: Repository, max_workers: int = 2):
        self.repo = repo
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="train")
        self._live: dict[str, TrainingRunner] = {}
        self._live_lock = threading.Lock()
        self._stopping = False
        self._admission_lock = threading.Lock()
        self._inflight_by_client: dict[str, int] = {}
        self._inflight_total = 0
        self._recent_by_client: dict[str, list[float]] = {}

    def shutdown(self, timeout: float = 20.0) -> None:
        """Stop accepting work and let in-flight jobs finish writing.

        wait=True matters: with wait=False the caller closes the SQLite
        connection out from under a still-running worker, whose next
        add_snapshot raises 'Cannot operate on a closed database' — and whose
        error handler then raises the same thing again, escaping _run and
        leaving the job row stuck at 'running'. That is the race that made
        every restart brick a visitor's training.

        _stopping short-circuits the training loop so wait=True returns
        promptly rather than blocking for the full budget.
        """
        self._stopping = True
        self._pool.shutdown(wait=True, cancel_futures=True)

    # ------------------------------------------------------------------ API
    def submit(
        self,
        session_id: str,
        variant: str,
        iterations: int,
        budget_seconds: float | None = None,
        client_key: str = "unknown",
    ) -> tuple[str, Future]:
        if variant not in GAME_REGISTRY:
            raise JobRejected(f"unknown variant {variant!r}")

        try:
            iterations = int(iterations)
        except (TypeError, ValueError):
            raise JobRejected("iterations must be a number") from None

        with self._admission_lock:
            # Three gates, in order of how easy they are to defeat.
            #
            # Per session: pure UX — stops one browser tab queueing on itself.
            # Trivially bypassed by discarding the cookie, so it defends nothing.
            if self.repo.session_has_active_job(session_id):
                raise JobRejected("a training job is already running for this session")

            # Per client IP: the gate that actually bounds abuse. Cookies are
            # free to mint; IPs are not. Behind Cloudflare this is CF-Connecting-IP.
            if self._inflight_by_client.get(client_key, 0) >= self.MAX_JOBS_PER_CLIENT:
                raise JobRejected("too many training jobs in flight from this client")

            # Global: the last line. The pool queue is a SimpleQueue with no
            # maxsize, so without this an attacker just queues work forever and
            # every REST caller parks on a future that will not run for hours.
            if self._inflight_total >= self.MAX_QUEUED_TOTAL:
                raise JobRejected("training queue is full, try again shortly")

            # Rolling window: bounds SUSTAINED load, which the concurrency gates
            # above cannot see (a serial caller frees its slot every time).
            now = time.monotonic()
            recent = [
                t for t in self._recent_by_client.get(client_key, [])
                if now - t < self.RATE_WINDOW_SECONDS
            ]
            if len(recent) >= self.MAX_JOBS_PER_WINDOW:
                self._recent_by_client[client_key] = recent
                raise JobRejected("training rate limit reached, try again in a minute")
            recent.append(now)
            self._recent_by_client[client_key] = recent

            # Bound the bookkeeping itself: an attacker rotating IPs must not
            # grow this dict without limit.
            if len(self._recent_by_client) > 4096:
                self._recent_by_client = {
                    k: v for k, v in self._recent_by_client.items()
                    if v and now - v[-1] < self.RATE_WINDOW_SECONDS
                }

            self._inflight_by_client[client_key] = (
                self._inflight_by_client.get(client_key, 0) + 1
            )
            self._inflight_total += 1

        iterations = max(1, min(self.MAX_ITERATIONS, iterations))
        budget = min(
            self.MAX_BUDGET_SECONDS,
            self.DEFAULT_BUDGET_SECONDS if budget_seconds is None else float(budget_seconds),
        )

        try:
            job_id = self.repo.create_job(session_id, variant, iterations, budget)
            future = self._pool.submit(
                self._run, job_id, session_id, variant, iterations, budget, client_key
            )
        except Exception:
            self._release(client_key)  # never leak an admission slot
            raise
        return job_id, future

    def _release(self, client_key: str) -> None:
        with self._admission_lock:
            remaining = self._inflight_by_client.get(client_key, 1) - 1
            if remaining > 0:
                self._inflight_by_client[client_key] = remaining
            else:
                self._inflight_by_client.pop(client_key, None)
            self._inflight_total = max(0, self._inflight_total - 1)

    def live_strategy(self, job_id: str) -> dict | None:
        """Current average strategy of a still-running job, for live charts.

        Read from the in-process runner, never persisted — snapshots stay
        metrics-only by design.
        """
        with self._live_lock:
            runner = self._live.get(job_id)
        return runner.get_strategy() if runner is not None else None

    # ------------------------------------------------------------------ run
    def _run(
        self,
        job_id: str,
        session_id: str,
        variant: str,
        iterations: int,
        budget: float,
        client_key: str = "unknown",
    ) -> None:
        # A queued job can start running after shutdown() has already drained
        # the pool and the caller has closed the connection. Bail before
        # touching the repo: writing here raises "Cannot operate on a closed
        # database", and the error handler then raises the same thing again,
        # escaping _run — which is exactly what stranded job rows at 'running'.
        if self._stopping:
            self._release(client_key)
            return

        self.repo.job_started(job_id)
        runner = TrainingRunner(game_factory=GAME_REGISTRY[variant], num_players=2)
        with self._live_lock:
            self._live[job_id] = runner

        done = 0
        try:
            start = time.monotonic()
            last_snapshot = 0.0

            while (
                done < iterations
                and (time.monotonic() - start) < budget
                and not self._stopping
            ):
                step = min(self.TRAIN_BATCH, iterations - done)
                runner.solver.train(step)
                done += step

                now = time.monotonic()
                if (now - last_snapshot) >= self.SNAPSHOT_MIN_INTERVAL or done >= iterations:
                    self.repo.add_snapshot(
                        job_id,
                        iteration=done,
                        exploitability=runner.solver.exploitability(),
                        game_values=runner.solver.expected_utility(),
                    )
                    last_snapshot = now

            if self._stopping:
                self._release(client_key)
                return

            self.repo.save_strategy(
                variant=variant,
                iterations=done,
                strategy=runner.get_strategy(),
                session_id=session_id,
                job_id=job_id,
            )
            self.repo.job_finished(job_id, done)
        except Repository.Closed:
            pass  # shutting down; reconcile_orphaned_jobs() fails the row at next boot
        except Exception as exc:  # noqa: BLE001 — job errors must land in the job row
            try:
                self.repo.job_finished(job_id, done, error=repr(exc))
            except Repository.Closed:
                pass
        finally:
            with self._live_lock:
                self._live.pop(job_id, None)
            self._release(client_key)
