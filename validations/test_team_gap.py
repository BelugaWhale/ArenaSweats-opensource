#!/usr/bin/env python3
import logging
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ranking_algorithm import (
    SIGMA_FLOOR,
    UNBALANCED_TEAM_MU_REDUCTION,
    _teammate_penalty_scale,
    _teammate_penalty_scale_gap_pct,
    _unbalanced_grace_reduction_pct,
    calculate_teammate_gap_modifiers,
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
        expected_random_scale = max(_teammate_penalty_scale_gap_pct(0.5), _teammate_penalty_scale(60, 30))
        self.assertAlmostEqual(scales["player"], expected_random_scale)
        self.assertEqual(grace_blocked, [False])

        gaps, scales, grace_blocked = self.calculate([34, 30], {"teammate_b"})
        self.assertAlmostEqual(gaps["player"], 1 - 34 / 60)
        self.assertAlmostEqual(scales["player"], _teammate_penalty_scale_gap_pct(1 - 34 / 60))
        self.assertEqual(grace_blocked, [True])

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
            success, _, _ = process_game_ratings(
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


if __name__ == "__main__":
    unittest.main()
