from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "scripts" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from near_dog_specialist_common import (  # noqa: E402
    apply_market_shrinkage,
    build_two_sided_candidates,
    summarize_signal,
)


class NearDogSpecialistTests(unittest.TestCase):
    def test_two_sided_candidates_invert_second_side(self):
        df = pd.DataFrame(
            [
                {
                    "tour": "atp",
                    "date": pd.Timestamp("2026-01-01"),
                    "date_key": "2026-01-01",
                    "month": "2026-01",
                    "player_a_name": "Favorite A",
                    "player_b_name": "Dog B",
                    "surface": "Hard",
                    "raw_prob_a": 0.65,
                    "artifact_prob_a": 0.64,
                    "market_prob_a": 0.60,
                    "odds_a": 1.40,
                    "actual_result": 1,
                    "elo_diff": 100.0,
                    "rank_diff": 50.0,
                }
            ]
        )

        candidates = build_two_sided_candidates(df, ["elo_diff", "rank_diff"])
        side_b = candidates.loc[candidates["side"].eq("B")].iloc[0]

        self.assertEqual(len(candidates), 2)
        self.assertEqual(side_b["player_name"], "Dog B")
        self.assertAlmostEqual(side_b["raw_prob"], 0.35)
        self.assertAlmostEqual(side_b["artifact_prob"], 0.36)
        self.assertAlmostEqual(side_b["market_prob"], 0.40)
        self.assertAlmostEqual(side_b["actual_result"], 0.0)
        self.assertAlmostEqual(side_b["elo_diff"], -100.0)
        self.assertAlmostEqual(side_b["rank_diff"], -50.0)
        self.assertTrue(side_b["is_near_dog"])

    def test_summarize_signal_filters_by_specialist_edge(self):
        df = pd.DataFrame(
            {
                "actual_result": [1, 0, 1],
                "odds": [2.5, 2.8, 2.4],
                "market_prob": [0.38, 0.36, 0.40],
                "specialist_prob": [0.42, 0.35, 0.43],
            }
        )

        summary = summarize_signal(df, "specialist_prob", edge_threshold=0.02)

        self.assertEqual(summary["bet_count"], 2)
        self.assertAlmostEqual(summary["hit_rate"], 1.0)
        self.assertGreater(summary["roi"], 0.0)

    def test_market_shrinkage_moves_model_prob_toward_market(self):
        model_prob = pd.Series([0.60, 0.30])
        market_prob = pd.Series([0.40, 0.40])

        shrunk = apply_market_shrinkage(model_prob, market_prob, 0.25)

        self.assertAlmostEqual(shrunk.iloc[0], 0.45)
        self.assertAlmostEqual(shrunk.iloc[1], 0.375)


if __name__ == "__main__":
    unittest.main()
