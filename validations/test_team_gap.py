#!/usr/bin/env python3
import logging
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ranking_algorithm import (
    FRESH_GAP_SATURATION,
    FRESH_GAP_TRIGGER,
    SIGMA_FLOOR,
    UNBALANCED_TEAM_MU_REDUCTION,
    _teammate_penalty_scale_gap_pct,
    _unbalanced_grace_reduction_pct,
    calculate_rating,
    calculate_teammate_gap_modifiers,
    check_for_unbalanced_lobby,
    instantiate_rating_model,
    process_game_ratings,
)
class TeamGapTests(unittest.TestCase):
    def setUp(self):
        self.model = instantiate_rating_model()

    def calculate(self, teammate_mus, repeated_ids):
        teams = [[self.model.rating(mu=60), *(self.model.rating(mu=mu) for mu in teammate_mus)]]
        player_ids = [["player", "teammate_b", "teammate_c"]]
        return calculate_teammate_gap_modifiers(
            teams,
            [True],
            player_ids,
            {
                "player": set(repeated_ids),
                "teammate_b": set(),
                "teammate_c": set(),
            },
        )

    def test_each_teammate_uses_own_curve_and_lowest_scale_wins(self):
        _, scales, grace_blocked = self.calculate([55, 30], {"teammate_b"})
        expected_random_scale = _teammate_penalty_scale_gap_pct(0.5, FRESH_GAP_TRIGGER, FRESH_GAP_SATURATION)
        self.assertAlmostEqual(scales["player"], expected_random_scale)
        self.assertEqual(grace_blocked, [False])

        gaps, scales, grace_blocked = self.calculate([34, 30], {"teammate_b"})
        self.assertAlmostEqual(gaps["player"], 1 - 34 / 60)
        self.assertAlmostEqual(scales["player"], _teammate_penalty_scale_gap_pct(1 - 34 / 60))
        self.assertEqual(grace_blocked, [True])

    def test_fresh_teammate_curve_uses_later_thresholds(self):
        self.assertEqual(_teammate_penalty_scale_gap_pct(FRESH_GAP_TRIGGER, FRESH_GAP_TRIGGER, FRESH_GAP_SATURATION), 1.0)
        self.assertEqual(_teammate_penalty_scale_gap_pct(FRESH_GAP_SATURATION, FRESH_GAP_TRIGGER, FRESH_GAP_SATURATION), 0.05)
        self.assertLess(_teammate_penalty_scale_gap_pct(FRESH_GAP_TRIGGER), 1.0)

    def test_grace_block_uses_repeated_teammates_own_gap(self):
        _, _, grace_blocked = self.calculate([40.21, 30], {"teammate_b"})
        self.assertEqual(grace_blocked, [False])

        _, _, grace_blocked = self.calculate([40.19, 55], {"teammate_b"})
        self.assertEqual(grace_blocked, [True])

    def test_process_blocks_grace_for_eligible_team(self):
        player_ids = ["player", "teammate_b", "teammate_c", "opponent_a", "opponent_b", "opponent_c"]
        ratings = {
            player_id: self.model.rating(mu=mu)
            for player_id, mu in zip(player_ids, [60, 40, 55, 25, 25, 25])
        }
        pregame_display = {player_id: calculate_rating(rating) for player_id, rating in ratings.items()}
        repeated_teammates = {player_id: set() for player_id in player_ids}
        repeated_teammates["player"] = {"teammate_b"}
        arena_format = {
            "name": "3x2",
            "team_count": 2,
            "team_size": 3,
            "player_count": 6,
            "placement_count": 2,
            "tophalf_cutoff": 1,
        }

        with patch("ranking_algorithm.check_for_unbalanced_lobby", return_value=(None, None)) as grace_check:
            success, updated_ratings, modifiers = process_game_ratings(
                self.model,
                [(player_id, 1 if index < 3 else 2) for index, player_id in enumerate(player_ids)],
                "test-game",
                ratings,
                logging.getLogger("test_team_gap"),
                {"player", "teammate_b"},
                arena_format=arena_format,
                repeated_teammate_ids_by_pid=repeated_teammates,
            )

        self.assertTrue(success)
        self.assertEqual(grace_check.call_args.kwargs["gm_team_eligible_mask"], [False, False])
        for player_id in player_ids:
            self.assertEqual(
                calculate_rating(updated_ratings[player_id]) - pregame_display[player_id],
                modifiers[player_id]["openskill_rating_change"]
                + modifiers[player_id]["unbalanced_grace_net"]
                + modifiers[player_id]["team_gap_net"]
                + modifiers[player_id]["protection_net"],
            )

    def test_unbalanced_grace_uses_continuous_tail_slope(self):
        self.assertAlmostEqual(_unbalanced_grace_reduction_pct(0.20), 0.20 * UNBALANCED_TEAM_MU_REDUCTION)
        self.assertAlmostEqual(_unbalanced_grace_reduction_pct(0.50), 0.189)
        self.assertLess(_unbalanced_grace_reduction_pct(0.50), 0.50 * UNBALANCED_TEAM_MU_REDUCTION)

    def test_process_never_reduces_sigma_below_floor(self):
        player_ids = [f"player_{index}" for index in range(24)]
        ratings = {player_id: self.model.rating(sigma=SIGMA_FLOOR + 0.0001) for player_id in player_ids}
        arena_format = {
            "name": "3x8",
            "team_count": 8,
            "team_size": 3,
            "player_count": 24,
            "placement_count": 8,
            "tophalf_cutoff": 4,
        }

        success, updated_ratings, _ = process_game_ratings(
            self.model,
            [(player_id, index // 3 + 1) for index, player_id in enumerate(player_ids)],
            "sigma-floor-test",
            ratings,
            logging.getLogger("test_team_gap"),
            set(player_ids),
            arena_format=arena_format,
        )

        self.assertTrue(success)
        for player_id in player_ids:
            self.assertAlmostEqual(updated_ratings[player_id].sigma, SIGMA_FLOOR)

    def test_solo_gm_without_repeated_teammates_gets_third_place_protection(self):
        player_ids = [f"p{index}" for index in range(18)]
        players = [(player_id, index // 3 + 1) for index, player_id in enumerate(player_ids)]
        arena_format = {
            "name": "3x6",
            "team_count": 6,
            "team_size": 3,
            "player_count": 18,
            "placement_count": 6,
            "tophalf_cutoff": 3,
        }

        def run(repeated_teammates):
            ratings = {
                player_id: self.model.rating(mu=60 if player_id == "p6" else 25, sigma=3)
                for player_id in player_ids
            }
            pregame_rating = calculate_rating(ratings["p6"])
            success, updated_ratings, modifiers = process_game_ratings(
                self.model,
                players,
                "third-place-protection",
                ratings,
                logging.getLogger("test_third_place_protection"),
                {"p6"},
                arena_format=arena_format,
                repeated_teammate_ids_by_pid={
                    player_id: ({"p7"} if player_id == "p6" and repeated_teammates else set())
                    for player_id in player_ids
                },
            )
            return success, pregame_rating, updated_ratings, modifiers

        success, pregame_rating, updated_ratings, modifiers = run(False)
        self.assertTrue(success)
        self.assertEqual(calculate_rating(updated_ratings["p6"]), pregame_rating)
        self.assertGreater(modifiers["p6"]["protection_net"], 0)
        self.assertTrue(any(modifiers[player_id]["protection_net"] < 0 for player_id in player_ids[9:]))

        success, pregame_rating, updated_ratings, modifiers = run(True)
        self.assertTrue(success)
        self.assertLess(calculate_rating(updated_ratings["p6"]), pregame_rating)
        self.assertEqual(modifiers["p6"]["protection_net"], 0)

    def test_afk_adjustments_are_included_in_exact_breakdown(self):
        player_ids = [f"p{index}" for index in range(6)]
        ratings = {player_id: self.model.rating(mu=25, sigma=3) for player_id in player_ids}
        pregame_ratings = {player_id: calculate_rating(rating) for player_id, rating in ratings.items()}
        success, updated_ratings, modifiers = process_game_ratings(
            self.model,
            [(player_id, 1 if index < 3 else 2) for index, player_id in enumerate(player_ids)],
            "afk-breakdown",
            ratings,
            logging.getLogger("test_afk_breakdown"),
            set(player_ids),
            afk_pids={"p0", "p3"},
            afk_protected_pids={"p4", "p5"},
            arena_format={
                "name": "3x2",
                "team_count": 2,
                "team_size": 3,
                "player_count": 6,
                "placement_count": 2,
                "tophalf_cutoff": 1,
            },
        )

        self.assertTrue(success)
        self.assertEqual(modifiers["p0"]["afk_penalty_applied"], 1)
        self.assertLess(modifiers["p0"]["protection_net"], 0)
        self.assertEqual(modifiers["p4"]["afk_protection_applied"], 1)
        self.assertGreater(modifiers["p4"]["protection_net"], 0)
        for player_id in player_ids:
            self.assertEqual(
                calculate_rating(updated_ratings[player_id]) - pregame_ratings[player_id],
                modifiers[player_id]["openskill_rating_change"]
                + modifiers[player_id]["unbalanced_grace_net"]
                + modifiers[player_id]["team_gap_net"]
                + modifiers[player_id]["protection_net"],
            )

    def test_grace_tilt_reallocates_complete_team_grace_by_mu_gap(self):
        arena_format = {
            "name": "3x6",
            "team_count": 6,
            "team_size": 3,
            "player_count": 18,
            "placement_count": 6,
            "tophalf_cutoff": 3,
        }
        stacked_mus = [52.0, 50.0, 48.0]
        stacked_sigmas = [3.5, 4.0, 4.5]
        player_ids = [f"p{index}" for index in range(18)]
        players = [(player_id, index // 3 + 1) for index, player_id in enumerate(player_ids)]
        logger = logging.getLogger("test_grace_tilt")

        def make_ratings():
            ratings = {}
            for index, player_id in enumerate(player_ids):
                mu = stacked_mus[index] if index < 3 else 38.0
                sigma = stacked_sigmas[index] if index < 3 else 4.0
                ratings[player_id] = self.model.rating(mu=mu, sigma=sigma)
            return ratings

        rate_input = [[make_ratings()[f"p{index}"] for index in range(team_index * 3, team_index * 3 + 3)] for team_index in range(6)]
        adjusted_teams, _ = check_for_unbalanced_lobby(self.model, rate_input, logger, gm_team_eligible_mask=[True] * 6)
        ordinary_output = self.model.rate(rate_input, ranks=list(range(6)))
        isolated_input = [[self.model.rating(mu=rating.mu, sigma=rating.sigma) for rating in (adjusted_teams[team_index] if team_index == 0 else rate_input[team_index])] for team_index in range(6)]
        graced_output = self.model.rate(isolated_input, ranks=list(range(6)))
        expected_team_mu_grace = sum(
            (graced_output[0][index].mu - isolated_input[0][index].mu)
            - (ordinary_output[0][index].mu - rate_input[0][index].mu)
            for index in range(3)
        )
        expected_team_sigma_grace = sum(
            (graced_output[0][index].sigma - isolated_input[0][index].sigma)
            - (ordinary_output[0][index].sigma - rate_input[0][index].sigma)
            for index in range(3)
        )

        with patch("ranking_algorithm.UNBALANCED_LOBBY_GRACE_ENABLED", False):
            success, ordinary_ratings, ordinary_modifiers = process_game_ratings(
                self.model,
                players,
                "grace-tilt-ordinary",
                make_ratings(),
                logger,
                set(player_ids),
                arena_format=arena_format,
            )
        self.assertTrue(success)

        tilted_start = make_ratings()
        tilted_pre_display = {player_id: calculate_rating(rating) for player_id, rating in tilted_start.items()}
        success, tilted_ratings, tilted_modifiers = process_game_ratings(
            self.model,
            players,
            "grace-tilt-graced",
            tilted_start,
            logger,
            set(player_ids),
            arena_format=arena_format,
        )
        self.assertTrue(success)
        self.assertGreater(tilted_modifiers["p0"]["unbalanced_reduction_pct"], 0.0)
        self.assertEqual(tilted_modifiers["p0"]["gap_scale"], 1.0)
        self.assertEqual(tilted_modifiers["p1"]["gap_scale"], 1.0)
        self.assertEqual(tilted_modifiers["p2"]["gap_scale"], 1.0)

        allocated_mu = []
        allocated_sigma = []
        for index in range(3):
            player_id = f"p{index}"
            allocated_mu.append(tilted_ratings[player_id].mu - ordinary_ratings[player_id].mu)
            allocated_sigma.append(tilted_ratings[player_id].sigma - ordinary_ratings[player_id].sigma)
        self.assertAlmostEqual(sum(allocated_mu), expected_team_mu_grace)
        self.assertAlmostEqual(sum(allocated_sigma), expected_team_sigma_grace)
        self.assertGreater(min(allocated_mu), 0.0)
        self.assertAlmostEqual(allocated_mu[2] / allocated_mu[0], stacked_mus[0] / stacked_mus[2])
        self.assertAlmostEqual(allocated_sigma[2] / allocated_sigma[0], stacked_mus[0] / stacked_mus[2])
        self.assertGreater(allocated_mu[2], allocated_mu[1])
        self.assertGreater(allocated_mu[1], allocated_mu[0])
        for index in range(3):
            player_id = f"p{index}"
            self.assertEqual(
                tilted_modifiers[player_id]["openskill_rating_change"],
                ordinary_modifiers[player_id]["openskill_rating_change"],
            )
            self.assertEqual(tilted_modifiers[player_id]["team_gap_net"], 0)
            self.assertEqual(
                tilted_modifiers[player_id]["unbalanced_grace_net"],
                calculate_rating(tilted_ratings[player_id]) - calculate_rating(ordinary_ratings[player_id]),
            )
            self.assertEqual(
                calculate_rating(tilted_ratings[player_id]) - tilted_pre_display[player_id],
                tilted_modifiers[player_id]["openskill_rating_change"]
                + tilted_modifiers[player_id]["unbalanced_grace_net"]
                + tilted_modifiers[player_id]["team_gap_net"]
                + tilted_modifiers[player_id]["protection_net"],
            )
        for index in range(3, 18):
            player_id = f"p{index}"
            self.assertEqual(tilted_modifiers[player_id]["unbalanced_reduction_pct"], 0.0)
            self.assertEqual(tilted_modifiers[player_id]["unbalanced_grace_net"], 0)
            self.assertEqual(
                tilted_modifiers[player_id]["openskill_rating_change"],
                calculate_rating(tilted_ratings[player_id]) - tilted_pre_display[player_id],
            )
        self.assertEqual(
            sum(tilted_modifiers[f"p{index}"]["unbalanced_grace_net"] for index in range(3)),
            sum(calculate_rating(tilted_ratings[f"p{index}"]) - calculate_rating(ordinary_ratings[f"p{index}"]) for index in range(3)),
        )

    def test_each_team_grace_is_isolated_from_other_adjusted_teams(self):
        raw_ratings = [
            [(39.4375889544, 2.8611140602), (39.4375889544, 2.8611140602), (38.5516419671, 3.4808456046)],
            [(53.6550337851, 2.5020304960), (41.6906775871, 2.5), (35.4261087949, 2.5595506175)],
            [(41.7328170389, 2.6358313129), (40.4051739273, 2.5006528636), (37.9930133482, 3.2295092480)],
            [(40.9064048812, 2.6772140686), (38.3491050838, 2.5), (36.6444985356, 2.5)],
            [(49.2069220422, 2.5614671720), (49.1723502736, 2.5256816008), (46.9626904941, 2.9401904225)],
            [(42.2839181973, 3.0411340658), (38.9886789874, 2.8645757395), (35.9179482565, 2.5034959732)],
        ]
        player_ids = [f"p{index}" for index in range(18)]
        players = [(player_id, index // 3 + 1) for index, player_id in enumerate(player_ids)]
        gm_set = {f"p{team_index * 3 + player_index}" for team_index in [1, 2, 4] for player_index in [0, 1]}
        arena_format = {"name": "3x6", "team_count": 6, "team_size": 3, "player_count": 18, "placement_count": 6, "tophalf_cutoff": 3}

        def make_ratings():
            return {
                player_ids[team_index * 3 + player_index]: self.model.rating(mu=mu, sigma=sigma)
                for team_index, team in enumerate(raw_ratings)
                for player_index, (mu, sigma) in enumerate(team)
            }

        with patch("ranking_algorithm.UNBALANCED_LOBBY_GRACE_ENABLED", False):
            success, _, ordinary_modifiers = process_game_ratings(
                self.model, players, "isolated-ordinary", make_ratings(), logging.getLogger("isolated_grace"), gm_set,
                arena_format=arena_format,
            )
        self.assertTrue(success)
        success, _, isolated_modifiers = process_game_ratings(
            self.model, players, "isolated-grace", make_ratings(), logging.getLogger("isolated_grace"), gm_set,
            arena_format=arena_format,
        )
        self.assertTrue(success)
        self.assertEqual(sum(isolated_modifiers[f"p{index}"]["unbalanced_grace_net"] for index in range(3, 6)), 22)
        for player_id in player_ids:
            self.assertEqual(isolated_modifiers[player_id]["openskill_rating_change"], ordinary_modifiers[player_id]["openskill_rating_change"])
        for index in list(range(3)) + list(range(9, 12)) + list(range(15, 18)):
            self.assertEqual(isolated_modifiers[f"p{index}"]["unbalanced_grace_net"], 0)


if __name__ == "__main__":
    unittest.main()
