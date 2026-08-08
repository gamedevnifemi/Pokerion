"""Round-trip tests for the SQLite repository."""

import pytest

from pokerion.server.repository import Repository


@pytest.fixture
def repo(tmp_path):
    r = Repository(tmp_path / "test.db")
    yield r
    r.close()


class TestMatchesAndHands:
    def test_match_and_hands_round_trip(self, repo):
        repo.touch_session("sid1")
        repo.create_match("m1", "sid1", "kuhn", 4)
        repo.record_hand(
            "m1",
            hand_index=0,
            human_seat=0,
            chips_delta=2.0,
            winner_seat=0,
            states=[{"is_terminal": True, "winner": 0}],
            strategy={"K": {"bet": 1.0, "check": 0.0}},
        )
        repo.update_match("m1", hand_index=1, human_chips=2.0, complete=False)

        matches = repo.list_matches_with_hands("sid1")
        assert len(matches) == 1
        m = matches[0]
        assert m["id"] == "m1"
        assert m["human_chips"] == 2.0
        assert not m["complete"]
        assert len(m["hands"]) == 1
        assert m["hands"][0]["chips_delta"] == 2.0
        assert m["hands"][0]["strategy"]["K"]["bet"] == 1.0

    def test_matches_are_session_scoped(self, repo):
        repo.touch_session("sid1")
        repo.touch_session("sid2")
        repo.create_match("m1", "sid1", "kuhn", 4)
        assert repo.list_matches_with_hands("sid2") == []


class TestJobs:
    def test_job_lifecycle(self, repo):
        job_id = repo.create_job("sid1", "kuhn", 1000, 10.0)
        assert repo.get_job(job_id)["status"] == "queued"
        assert repo.session_has_active_job("sid1")

        repo.job_started(job_id)
        assert repo.get_job(job_id)["status"] == "running"

        repo.job_finished(job_id, 1000)
        job = repo.get_job(job_id)
        assert job["status"] == "done"
        assert job["iterations_done"] == 1000
        assert not repo.session_has_active_job("sid1")

    def test_job_error_recorded(self, repo):
        job_id = repo.create_job("sid1", "kuhn", 1000, 10.0)
        repo.job_started(job_id)
        repo.job_finished(job_id, 400, error="RuntimeError('boom')")
        job = repo.get_job(job_id)
        assert job["status"] == "error"
        assert "boom" in job["error"]

    def test_snapshots_stream_incrementally(self, repo):
        job_id = repo.create_job("sid1", "kuhn", 1000, 10.0)
        repo.add_snapshot(job_id, 100, 0.5, [0.1, -0.1])
        repo.add_snapshot(job_id, 200, 0.3, [0.05, -0.05])

        first = repo.snapshots_after(job_id)
        assert [s["iteration"] for s in first] == [100, 200]

        # Tailing: nothing new after the last seen id
        assert repo.snapshots_after(job_id, first[-1]["id"]) == []


class TestStrategies:
    def test_reference_fallback(self, repo):
        assert not repo.has_reference("kuhn")
        repo.save_strategy("kuhn", 25_000, {"K": {"bet": 0.8, "check": 0.2}})
        assert repo.has_reference("kuhn")

        row = repo.latest_strategy("kuhn", "some-session")
        assert row["source"] == "reference"
        assert row["payload"]["K"]["bet"] == 0.8

    def test_session_strategy_wins_over_reference(self, repo):
        repo.save_strategy("kuhn", 25_000, {"K": {"bet": 0.8, "check": 0.2}})
        repo.save_strategy(
            "kuhn", 500, {"K": {"bet": 0.5, "check": 0.5}}, session_id="sid1", job_id="j1"
        )

        mine = repo.latest_strategy("kuhn", "sid1")
        assert mine["source"] == "session"
        assert mine["iterations"] == 500

        theirs = repo.latest_strategy("kuhn", "other")
        assert theirs["source"] == "reference"

    def test_no_strategy_at_all(self, repo):
        assert repo.latest_strategy("kuhn", None) is None
