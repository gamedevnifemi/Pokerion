"""End-to-end API smoke tests over the real app (lifespan, SQLite, executor)."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os

    os.environ["POKERION_DB"] = str(tmp_path_factory.mktemp("db") / "api.db")
    from pokerion.server.app import app

    # Context manager runs the lifespan: opens the repo, bakes the reference.
    with TestClient(app) as c:
        yield c


class TestSessionCookie:
    def test_first_request_sets_session_cookie(self, client):
        response = client.get("/api/games")
        assert response.status_code == 200
        assert "pokerion_sid" in client.cookies


class TestMatchFlow:
    def test_full_hand_and_persistence(self, client):
        created = client.post("/api/match/new?variant=kuhn&length=2").json()
        match_id = created["match_id"]
        state = created["state"]
        assert state["match"]["length"] == 2
        # Play works on arrival: the agent follows the baked reference strategy.

        # Drive one hand to completion
        terminal = None
        for _ in range(6):
            action = "check" if "check" in state["legal_actions"] else "call"
            result = client.post(f"/api/match/{match_id}/action?action={action}").json()
            assert "error" not in result
            state = result["state"]
            if result.get("terminal"):
                terminal = result
                break
        assert terminal is not None
        assert terminal["hand_meta"]["human_seat"] == 0
        assert len([s for s in terminal["replay"] if s["is_terminal"]]) == 1

        # The hand is in the store, scoped to this session
        sessions = client.get("/api/replay/sessions").json()["sessions"]
        assert any(m["id"] == match_id and len(m["hands"]) == 1 for m in sessions)

    def test_illegal_action_is_a_400_not_corruption(self, client):
        created = client.post("/api/match/new?variant=kuhn&length=2").json()
        match_id = created["match_id"]
        response = client.post(f"/api/match/{match_id}/action?action=fold")
        assert response.status_code == 400
        assert "error" in response.json()

    def test_unknown_match_is_404(self, client):
        assert client.post("/api/match/nope/action?action=check").status_code == 404


class TestTraining:
    def test_rest_training_runs_a_job(self, client):
        result = client.post("/api/train?iterations=300&variant=kuhn").json()
        assert result["total_iterations"] == 300
        assert len(result["snapshots"]) >= 1
        assert result["strategy"]  # final strategy present
        # Snapshots are metrics only — the schema has no strategy column
        assert "strategy" not in result["snapshots"][0]

    def test_strategy_endpoint_prefers_session_training(self, client):
        # Previous test trained 300 iterations under this same session cookie
        result = client.get("/api/strategy?variant=kuhn").json()
        assert result["source"] == "session"
        assert result["iterations"] == 300

    def test_ws_training_streams_and_completes(self, client):
        with client.websocket_connect("/api/ws/train") as ws:
            ws.send_json({"iterations": 300, "variant": "kuhn"})
            saw_progress = False
            while True:
                msg = ws.receive_json()
                if msg["type"] == "progress":
                    saw_progress = True
                    assert "exploitability" in msg
                if msg["type"] == "done":
                    assert msg["iteration"] == 300
                    assert msg["strategy"]
                    break
                assert msg["type"] != "error", msg
            assert saw_progress
