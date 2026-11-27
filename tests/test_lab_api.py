"""Tests for the lab API: the endpoint behind the Lab page."""

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pokerion.server.routes import lab


@pytest.fixture(scope="module")
def client():
    # The lab routes on their own. They need neither the database nor the
    # session cookie, and the real app's database is opened and closed by
    # other test files, which would leak into these tests.
    app = FastAPI()
    app.include_router(lab.router)
    return TestClient(app)


def test_the_site_serves_the_lab_routes():
    from pokerion.server.app import app

    paths = {route.path for route in app.routes}
    assert {"/api/lab/options", "/api/lab/match"} <= paths


@pytest.fixture
def lab_on(monkeypatch):
    monkeypatch.setenv("POKERION_LAB", "1")


def table(*kinds, **options):
    return {"seats": [{"kind": kind} for kind in kinds], **options}


class TestSwitch:
    def test_off_unless_enabled(self, client, monkeypatch):
        monkeypatch.delenv("POKERION_LAB", raising=False)
        assert client.get("/api/lab/options").status_code == 404
        response = client.post("/api/lab/match", json=table("rock", "maniac", game="kuhn-2p"))
        assert response.status_code == 404

    def test_on_when_enabled(self, client, lab_on):
        assert client.get("/api/lab/options").status_code == 200


class TestOptions:
    def test_lists_games_players_and_limits(self, client, lab_on):
        options = client.get("/api/lab/options").json()
        games = {game["id"]: game for game in options["games"]}
        assert games["kuhn-2p"]["players"] == 2
        assert games["kuhn-3p"]["players"] == 3
        assert len(games["kuhn-3p"]["table"]) == 3
        assert options["max_hands"] >= 1000

    def test_every_default_table_uses_known_kinds(self, client, lab_on):
        options = client.get("/api/lab/options").json()
        kinds = {kind["id"] for kind in options["kinds"]}
        for game in options["games"]:
            assert set(game["table"]) <= kinds
        assert [k["id"] for k in options["kinds"] if k["adapts"]] == ["adaptive"]


class TestMatch:
    @pytest.fixture(scope="class")
    def match(self, client):
        import os

        os.environ["POKERION_LAB"] = "1"
        try:
            body = table("adaptive", "rock", "maniac", game="kuhn-3p", hands=60, seed=3)
            response = client.post("/api/lab/match", json=body)
        finally:
            del os.environ["POKERION_LAB"]
        assert response.status_code == 200
        return response.json()

    def test_names_the_table(self, match):
        assert [agent["name"] for agent in match["agents"]] == ["Adapting agent", "Rock", "Maniac"]
        assert [agent["adapts"] for agent in match["agents"]] == [True, False, False]
        assert match["game"]["players"] == 3
        assert len(match["game"]["infosets"]) == 48

    def test_scores_every_hand_and_every_hand_is_zero_sum(self, match):
        per_hand = np.array(match["per_hand"])
        assert per_hand.shape == (3, 60)
        assert np.abs(per_hand.sum(axis=0)).max() < 1e-4
        assert match["winnings"] == pytest.approx(per_hand.mean(axis=1), abs=1e-4)

    def test_describes_every_dealt_hand(self, match):
        assert len(match["plays"]) == 60
        for play in match["plays"]:
            assert sorted(play["seating"]) == [0, 1, 2]
            assert len(play["cards"]) == 3 and len(set(play["cards"])) == 3
            assert play["pot"] == sum(play["committed"])
            assert set(play["shown"]) <= {0, 1, 2}
            assert play["winner"] in (0, 1, 2)
            assert len(play["expected"]) == 3

    def test_a_dealt_hand_follows_the_bots_at_the_table(self, match):
        # The rock never bets or calls and the maniac never checks or folds, wherever they sit.
        for play in match["plays"]:
            for seat, action in play["actions"]:
                kind = match["agents"][play["seating"][seat]]["kind"]
                if kind == "rock":
                    assert action in ("check", "fold")
                if kind == "maniac":
                    assert action in ("bet", "call")

    def test_every_card_is_shown_when_the_match_turns_them_all_over(self, match):
        assert match["reveal"] == "all"
        for play in match["plays"]:
            assert play["shown"] == [0, 1, 2]
            assert set(play["showdown"]) <= {0, 1, 2}

    def test_a_fold_stays_hidden_when_only_showdowns_are_shown(self, client, lab_on):
        body = table("maniac", "rock", game="kuhn-2p", hands=10, reveal="showdown")
        plays = client.post("/api/lab/match", json=body).json()["plays"]
        # The rock folds to every bet, so nothing ever reaches a showdown.
        assert all(play["showdown"] == [] and play["shown"] == [] for play in plays)

    def test_gives_the_adapting_agent_its_two_reference_lines(self, match):
        assert set(match["references"]) == {"0"}
        reference = match["references"]["0"]
        assert reference["fixed"] == pytest.approx(0.16, abs=0.01)   # solid play at this table
        assert reference["ceiling"] == pytest.approx(0.75, abs=0.01)  # the most possible
        assert reference["fixed"] < reference["ceiling"]

    def test_records_what_the_adapting_agent_believed(self, match):
        beliefs = match["beliefs"]["0"]
        assert beliefs["hands_seen"][0] == 0 and beliefs["hands_seen"][-1] == 60
        assert set(beliefs["about"]) == {"1", "2"}
        about_rock = beliefs["about"]["1"]
        assert len(about_rock["predicted"]) == len(beliefs["hands_seen"])
        assert len(about_rock["predicted"][0]) == 48
        assert all(sum(row) == pytest.approx(1.0, abs=0.01) for row in about_rock["predicted"][-1])
        assert len(about_rock["actual"]) == 48

    def test_beliefs_about_the_rock_move_towards_the_truth(self, match):
        about_rock = match["beliefs"]["0"]["about"]["1"]
        actual = np.array(about_rock["actual"])
        before = np.abs(np.array(about_rock["predicted"][0]) - actual).mean()
        after = np.abs(np.array(about_rock["predicted"][-1]) - actual).mean()
        assert after < before

    def test_the_same_seed_gives_the_same_match(self, client, lab_on):
        body = table("adaptive", "random", game="kuhn-2p", hands=20, seed=7)
        first = client.post("/api/lab/match", json=body).json()
        second = client.post("/api/lab/match", json=body).json()
        assert first["per_hand"] == second["per_hand"]
        assert first["plays"] == second["plays"]

    def test_two_of_a_kind_get_numbered_names(self, client, lab_on):
        body = table("rock", "rock", "maniac", game="kuhn-3p", hands=6)
        agents = client.post("/api/lab/match", json=body).json()["agents"]
        assert [agent["name"] for agent in agents] == ["Rock 1", "Rock 2", "Maniac"]

    def test_no_references_when_two_agents_adapt_to_each_other(self, client, lab_on):
        body = table("adaptive", "adaptive", game="kuhn-2p", hands=10)
        match = client.post("/api/lab/match", json=body).json()
        assert match["references"] == {}
        assert match["beliefs"]["0"]["about"]["1"]["actual"] is None

    def test_an_agent_with_no_trust_plays_like_solid_play(self, client, lab_on):
        seats = [{"kind": "adaptive", "trust": 0.0}, {"kind": "rock"}]
        match = client.post(
            "/api/lab/match", json={"game": "kuhn-2p", "seats": seats, "hands": 20}
        ).json()
        assert match["winnings"][0] == pytest.approx(match["references"]["0"]["fixed"], abs=1e-4)


class TestBadRequests:
    def test_unknown_game(self, client, lab_on):
        response = client.post("/api/lab/match", json=table("rock", "rock", game="holdem"))
        assert response.status_code == 404
        assert "unknown game" in response.json()["error"]

    def test_wrong_number_of_seats(self, client, lab_on):
        response = client.post("/api/lab/match", json=table("rock", "rock", game="kuhn-3p"))
        assert response.status_code == 400
        assert "3 seats" in response.json()["error"]

    def test_unknown_kind_of_player(self, client, lab_on):
        response = client.post("/api/lab/match", json=table("rock", "wizard", game="kuhn-2p"))
        assert response.status_code == 400
        assert "wizard" in response.json()["error"]

    def test_too_many_hands(self, client, lab_on):
        body = table("rock", "rock", game="kuhn-2p", hands=10**6)
        assert client.post("/api/lab/match", json=body).status_code == 422
