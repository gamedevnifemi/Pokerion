"""Tests for the compiled game tree."""

import numpy as np
import pytest

from pokerion.game.kuhn import KUHN_2P, KUHN_3P
from pokerion.game.tree import CHANCE, GameTree
from tests.helpers import BiasedCoin, random_strategy, walk_expected_utilities


class TestShape:
    def test_two_player_kuhn(self, kuhn):
        # The root, then for each of 6 deals: 4 decision nodes and 5 endings.
        assert kuhn.num_nodes == 1 + 6 * 9
        assert kuhn.num_infosets == 12
        assert kuhn.num_players == 2
        assert kuhn.variant == "kuhn"
        assert set(kuhn.actions) == {"check", "bet", "call", "fold"}

    def test_three_player_kuhn(self, kuhn3):
        # The root, then for each of 24 deals: 12 decision nodes and 13 endings.
        assert kuhn3.num_nodes == 1 + 24 * 25
        assert kuhn3.num_infosets == 48
        assert np.bincount(kuhn3.infoset_player).tolist() == [16, 16, 16]

    def test_legal_actions_match_the_game(self, kuhn):
        row = kuhn.row("K|check:bet")
        legal = {a for a in kuhn.actions if kuhn.legal[row, kuhn.column(a)]}
        assert legal == {"call", "fold"}
        assert kuhn.infoset_actions[row] == ("call", "fold")
        assert kuhn.infoset_player[row] == 0


class TestLayout:
    def test_children_are_numbered_after_their_parents(self, kuhn3):
        assert (kuhn3.edge_child > kuhn3.edge_parent).all()
        assert (np.diff(kuhn3.edge_child) == 1).all()  # breadth-first: one new node per edge

    def test_levels_tile_the_edges_in_order(self, kuhn3):
        stops = [level.stop for level in kuhn3.levels]
        starts = [level.start for level in kuhn3.levels]
        assert starts == [0] + stops[:-1]
        assert stops[-1] == len(kuhn3.edge_parent)

    def test_a_level_never_feeds_itself(self, kuhn3):
        for level in kuhn3.levels:
            parents = set(kuhn3.edge_parent[level].tolist())
            children = set(kuhn3.edge_child[level].tolist())
            assert not parents & children

    def test_first_level_is_the_deal(self, kuhn):
        deal = kuhn.levels[0]
        assert (kuhn.edge_owner[deal] == CHANCE).all()
        assert kuhn.edge_chance_prob[deal].sum() == pytest.approx(1.0)

    def test_payoffs_only_at_endings(self, kuhn3):
        has_children = np.zeros(kuhn3.num_nodes, dtype=bool)
        has_children[kuhn3.edge_parent] = True
        assert (kuhn3.utilities[has_children] == 0).all()
        assert (kuhn3.utilities[~has_children].sum(axis=1) == 0).all()  # zero-sum


class TestStrategies:
    def test_uniform_strategy_is_valid(self, kuhn3):
        kuhn3.check_strategy(kuhn3.uniform_strategy())

    def test_dict_round_trip(self, kuhn3):
        strategy = random_strategy(kuhn3, np.random.default_rng(0))
        again = kuhn3.strategy_from_dict(kuhn3.strategy_to_dict(strategy))
        assert np.allclose(again, strategy)

    def test_dict_lists_legal_actions_in_game_order(self, kuhn):
        as_dict = kuhn.strategy_to_dict(kuhn.uniform_strategy())
        assert list(as_dict["K"]) == ["check", "bet"]
        assert list(as_dict["K|bet"]) == ["call", "fold"]

    def test_missing_information_sets_are_played_uniformly(self, kuhn):
        strategy = kuhn.strategy_from_dict({"K": {"check": 0.0, "bet": 1.0}})
        assert strategy[kuhn.row("K"), kuhn.column("bet")] == 1.0
        assert strategy[kuhn.row("Q"), kuhn.column("bet")] == 0.5

    def test_rejects_the_wrong_shape(self, kuhn):
        with pytest.raises(ValueError, match="shape"):
            kuhn.check_strategy(np.ones((3, 3)))

    def test_rejects_negative_probabilities(self, kuhn):
        strategy = kuhn.uniform_strategy()
        row = kuhn.row("K")
        strategy[row, kuhn.column("check")] = 1.5
        strategy[row, kuhn.column("bet")] = -0.5
        with pytest.raises(ValueError, match="negative"):
            kuhn.check_strategy(strategy)

    def test_rejects_probability_on_an_illegal_action(self, kuhn):
        strategy = kuhn.uniform_strategy()
        strategy[kuhn.row("K"), kuhn.column("fold")] = 0.1
        with pytest.raises(ValueError, match="illegal"):
            kuhn.check_strategy(strategy)

    def test_rejects_rows_that_do_not_sum_to_one(self, kuhn):
        strategy = kuhn.uniform_strategy()
        strategy[kuhn.row("Q")] *= 0.5
        with pytest.raises(ValueError, match="'Q'"):
            kuhn.check_strategy(strategy)

    def test_combine_takes_each_row_from_the_seat_that_acts_there(self, kuhn3):
        rng = np.random.default_rng(1)
        seats = [random_strategy(kuhn3, rng) for _ in range(3)]
        profile = kuhn3.combine(seats)
        for row, seat in enumerate(kuhn3.infoset_player):
            assert np.array_equal(profile[row], seats[seat][row])

    def test_combine_needs_one_strategy_per_seat(self, kuhn3):
        with pytest.raises(ValueError, match="need 3"):
            kuhn3.combine([kuhn3.uniform_strategy()] * 2)


class TestPasses:
    @pytest.mark.parametrize("rules", [KUHN_2P, KUHN_3P])
    def test_expected_utilities_match_a_plain_walk_of_the_game(self, rules):
        tree = GameTree(rules.root())
        rng = np.random.default_rng(2)
        for _ in range(5):
            strategy = random_strategy(tree, rng)
            reference = walk_expected_utilities(rules.root(), tree.strategy_to_dict(strategy))
            assert tree.expected_utilities(strategy) == pytest.approx(reference, abs=1e-12)

    def test_uniform_play_in_kuhn(self, kuhn):
        # Worked by hand: the first player wins 1/8 of a chip per hand when both play at random.
        assert kuhn.expected_utilities(kuhn.uniform_strategy()) == pytest.approx([0.125, -0.125])

    def test_root_value_is_the_expected_utility(self, kuhn3):
        strategy = random_strategy(kuhn3, np.random.default_rng(3))
        values = kuhn3.node_values(strategy)
        assert values.shape == (kuhn3.num_nodes, 3)
        assert values[0] == pytest.approx(kuhn3.expected_utilities(strategy))

    def test_endings_are_reached_with_total_probability_one(self, kuhn3):
        reach = kuhn3.reach(random_strategy(kuhn3, np.random.default_rng(4)))
        is_ending = np.ones(kuhn3.num_nodes, dtype=bool)
        is_ending[kuhn3.edge_parent] = False
        assert reach.total()[is_ending].sum() == pytest.approx(1.0)

    def test_leaving_a_seat_out_removes_exactly_its_factor(self, kuhn3):
        reach = kuhn3.reach(random_strategy(kuhn3, np.random.default_rng(5)))
        for seat in range(3):
            assert np.allclose(reach.without(seat) * reach.players[seat], reach.total())

    def test_chance_that_is_not_uniform_is_weighted_correctly(self):
        tree = GameTree(BiasedCoin())
        always_heads = tree.strategy_from_dict({"guess": {"h": 1.0, "t": 0.0}})
        assert tree.expected_utilities(always_heads) == pytest.approx([0.4, -0.4])

    def test_sum_by_slot_groups_edges_by_information_set_and_action(self, kuhn):
        player_edges = kuhn.edge_owner != CHANCE
        counts = kuhn.sum_by_slot(player_edges, np.ones(player_edges.sum()))
        # Each information set holds two deals (the opponent's two possible cards).
        assert (counts[kuhn.legal] == 2).all()
        assert (counts[~kuhn.legal] == 0).all()


class TestNodes:
    def test_each_nodes_edges_are_one_contiguous_run(self, kuhn3):
        for node in range(kuhn3.num_nodes):
            first, count = kuhn3.node_first_edge[node], kuhn3.node_edge_count[node]
            assert (kuhn3.edge_parent[first:first + count] == node).all()
        assert kuhn3.node_edge_count.sum() == len(kuhn3.edge_parent)

    def test_endings_have_no_edges(self, kuhn):
        is_ending = np.ones(kuhn.num_nodes, dtype=bool)
        is_ending[kuhn.edge_parent] = False
        assert (kuhn.node_edge_count[is_ending] == 0).all()
        assert (kuhn.node_edge_count[~is_ending] > 0).all()

    def test_each_information_set_names_one_of_its_own_nodes(self, kuhn3):
        for row, node in enumerate(kuhn3.infoset_node):
            first = kuhn3.node_first_edge[node]
            assert kuhn3.edge_slot[first] // kuhn3.num_actions == row
            assert kuhn3.edge_owner[first] == kuhn3.infoset_player[row]

    def test_showdown_marks_the_seats_that_show_their_cards(self, kuhn):
        is_ending = kuhn.node_edge_count == 0
        shown = kuhn.showdown[is_ending].sum(axis=1)
        # Of the 5 ways a hand ends, 2 are folds (nobody shows) and 3 are showdowns (both show).
        assert sorted(np.unique(shown).tolist()) == [0, 2]
        assert (shown == 2).sum() == 6 * 3
        assert not kuhn.showdown[~is_ending].any()


class TestSamplePlay:
    def test_a_played_hand_runs_from_the_root_to_an_ending(self, kuhn3):
        rng = np.random.default_rng(6)
        strategy = random_strategy(kuhn3, rng)
        for _ in range(50):
            played = kuhn3.sample_play(strategy, rng)
            assert kuhn3.edge_parent[played[0]] == 0
            assert (kuhn3.edge_child[played[:-1]] == kuhn3.edge_parent[played[1:]]).all()
            assert kuhn3.node_edge_count[kuhn3.edge_child[played[-1]]] == 0

    def test_endings_come_up_as_often_as_their_reach_says(self, kuhn):
        rng = np.random.default_rng(7)
        strategy = random_strategy(kuhn, rng)
        seen = np.zeros(kuhn.num_nodes)
        samples = 6000
        for _ in range(samples):
            seen[kuhn.edge_child[kuhn.sample_play(strategy, rng)[-1]]] += 1
        expected = kuhn.reach(strategy).total() * (kuhn.node_edge_count == 0)
        assert np.abs(seen / samples - expected).max() < 0.02

    def test_never_takes_an_action_with_zero_probability(self, kuhn):
        rng = np.random.default_rng(8)
        always_check_or_fold = kuhn.strategy_from_dict(
            {key: {a: float(a in ("check", "fold")) for a in kuhn.infoset_actions[kuhn.row(key)]}
             for key in kuhn.infoset_keys}
        )
        for _ in range(200):
            played = kuhn.sample_play(always_check_or_fold, rng)
            taken = [kuhn.actions[kuhn.edge_slot[e] % kuhn.num_actions]
                     for e in played if kuhn.edge_owner[e] != CHANCE]
            assert taken == ["check", "check"]


class TestMixture:
    def test_endpoints_play_like_the_originals(self, kuhn3):
        rng = np.random.default_rng(9)
        first, second = random_strategy(kuhn3, rng), random_strategy(kuhn3, rng)
        assert np.allclose(kuhn3.mixture(first, second, 0.0), first)
        assert np.allclose(kuhn3.mixture(first, second, 1.0), second)

    def test_mixture_is_a_valid_strategy(self, kuhn3):
        rng = np.random.default_rng(10)
        mixed = kuhn3.mixture(random_strategy(kuhn3, rng), random_strategy(kuhn3, rng), 0.3)
        kuhn3.check_strategy(mixed)

    @pytest.mark.parametrize("seat", [0, 1, 2])
    def test_payoffs_mix_in_the_same_proportion(self, kuhn3, seat):
        # Choosing one of two strategies per hand mixes every seat's expected payoff linearly.
        rng = np.random.default_rng(11 + seat)
        base, other = random_strategy(kuhn3, rng), random_strategy(kuhn3, rng)
        rows = kuhn3.infoset_player == seat

        def with_seat_playing(strategy):
            profile = base.copy()
            profile[rows] = strategy[rows]
            return kuhn3.expected_utilities(profile)

        weight = 0.3
        mixed = with_seat_playing(kuhn3.mixture(base, other, weight))
        expected = (1 - weight) * with_seat_playing(base) + weight * with_seat_playing(other)
        assert mixed == pytest.approx(expected, abs=1e-12)

    def test_it_is_not_the_same_as_averaging_each_decision(self, kuhn):
        always_bet = kuhn.strategy_from_dict(
            {key: {a: float(a in ("bet", "call")) for a in kuhn.infoset_actions[kuhn.row(key)]}
             for key in kuhn.infoset_keys}
        )
        never_bet = kuhn.strategy_from_dict(
            {key: {a: float(a in ("check", "fold")) for a in kuhn.infoset_actions[kuhn.row(key)]}
             for key in kuhn.infoset_keys}
        )
        mixed = kuhn.mixture(never_bet, always_bet, 0.5)
        # After checking first, the player must be the one who never bets, so it folds to a bet.
        assert mixed[kuhn.row("K|check:bet"), kuhn.column("fold")] == pytest.approx(1.0)
        averaged = 0.5 * never_bet + 0.5 * always_bet
        assert averaged[kuhn.row("K|check:bet"), kuhn.column("fold")] == 0.5

    def test_rejects_a_weight_outside_zero_to_one(self, kuhn):
        with pytest.raises(ValueError, match="between 0 and 1"):
            kuhn.mixture(kuhn.uniform_strategy(), kuhn.uniform_strategy(), -0.1)


class TestGuards:
    def test_refuses_a_game_that_is_too_large(self):
        with pytest.raises(ValueError, match="too large"):
            GameTree(KUHN_3P.root(), max_nodes=100)

    def test_refuses_an_information_set_with_inconsistent_actions(self):
        with pytest.raises(ValueError, match="different legal actions"):
            GameTree(BiasedCoin(lopsided=True))
