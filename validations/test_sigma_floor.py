import logging
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ranking_algorithm as ranking


class SigmaFloorTests(unittest.TestCase):
    def test_grace_redistribution_respects_floor_and_exact_breakdown(self):
        for team_size, team_count, sigma in [(3, 6, 2.501), (2, 8, 2.502)]:
            with self.subTest(team_size=team_size), patch.multiple(
                ranking,
                IS_3X6=team_size == 3,
                UNBALANCED_TEAM_MU_REDUCTION=0.57 if team_size == 3 else 0.22,
                UNBALANCED_PAIR_RATIO_ALPHA=2.5 if team_size == 3 else 3.0,
            ):
                model = ranking.instantiate_rating_model()
                players = [(i, i // team_size + 1) for i in range(team_size * team_count)]
                ratings = {
                    i: model.rating(mu=45 - i if i < team_size else 35, sigma=sigma if i < team_size else ranking.SIGMA_FLOOR)
                    for i, _ in players
                }
                before = {i: ranking.calculate_rating(rating) for i, rating in ratings.items()}
                success, updated, modifiers = ranking.process_game_ratings(
                    model, players, "grace-sigma-floor", ratings,
                    logging.getLogger("test_sigma_floor"), set(ratings),
                    arena_format={
                        "name": f"{team_size}x{team_count}", "team_count": team_count,
                        "team_size": team_size, "player_count": len(players),
                        "placement_count": team_count, "tophalf_cutoff": team_count // 2,
                    },
                    repeated_teammate_ids_by_pid={i: set() for i, _ in players},
                )

                self.assertTrue(success)
                self.assertGreater(modifiers[team_size - 1]["unbalanced_reduction_pct"], 0)
                self.assertEqual(modifiers[team_size - 1]["gap_scale"], 1)
                self.assertEqual(updated[team_size - 1].sigma, ranking.SIGMA_FLOOR)
                self.assertGreater(updated[0].sigma, ranking.SIGMA_FLOOR)
                for i, _ in players:
                    self.assertGreaterEqual(updated[i].sigma, ranking.SIGMA_FLOOR)
                    self.assertEqual(
                        ranking.calculate_rating(updated[i]) - before[i],
                        sum(modifiers[i][key] for key in ["openskill_rating_change", "unbalanced_grace_net", "team_gap_net", "protection_net"]),
                    )


if __name__ == "__main__":
    unittest.main()
