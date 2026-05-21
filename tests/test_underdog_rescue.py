from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "scripts" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from underdog_rescue import (  # noqa: E402
    assign_odds_bucket,
    assign_walk_forward_periods,
    select_candidate_on_tune,
    summarize_rows,
)


class UnderdogRescueTests(unittest.TestCase):
    def test_assign_odds_bucket_boundaries(self):
        odds = pd.Series([1.79, 1.80, 2.20, 2.21, None])

        buckets = assign_odds_bucket(odds)

        self.assertEqual(buckets.tolist(), ["favorite", "balanced", "balanced", "underdog", "unknown"])

    def test_summarize_rows_uses_flat_stake_roi_and_edge_filter(self):
        df = pd.DataFrame(
            {
                "actual_result": [1, 0, 1],
                "odds_a": [3.0, 4.0, 2.5],
                "market_prob_a": [0.30, 0.25, 0.40],
                "model_prob": [0.35, 0.20, 0.42],
            }
        )

        summary = summarize_rows(df, "model_prob", edge_threshold=0.0)

        self.assertEqual(summary["bet_count"], 2)
        self.assertAlmostEqual(summary["roi"], ((3.0 - 1.0) + (2.5 - 1.0)) / 2)
        self.assertAlmostEqual(summary["actual_win_rate"], 1.0)
        self.assertAlmostEqual(summary["mean_clv"], ((0.35 - 0.30) + (0.42 - 0.40)) / 2)

    def test_walk_forward_periods_are_ordered_and_non_overlapping(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=10, freq="D"),
                "value": range(10),
            }
        )

        out = assign_walk_forward_periods(df, train_fraction=0.50, tune_fraction=0.30)

        self.assertEqual(set(out["wf_period"]), {"train", "tune", "final"})
        self.assertLess(
            out.loc[out["wf_period"].eq("train"), "date"].max(),
            out.loc[out["wf_period"].eq("tune"), "date"].min(),
        )
        self.assertLess(
            out.loc[out["wf_period"].eq("tune"), "date"].max(),
            out.loc[out["wf_period"].eq("final"), "date"].min(),
        )

    def test_select_candidate_uses_tune_period_only(self):
        experiments = pd.DataFrame(
            [
                {
                    "candidate": "bad_final_good_tune",
                    "wf_period": "tune",
                    "selectable": True,
                    "status": "OK",
                    "bet_count": 30,
                    "roi": 0.05,
                    "mean_clv": 0.01,
                    "calibration_error": 0.03,
                },
                {
                    "candidate": "great_final_bad_tune",
                    "wf_period": "tune",
                    "selectable": True,
                    "status": "OK",
                    "bet_count": 30,
                    "roi": -0.10,
                    "mean_clv": 0.05,
                    "calibration_error": 0.02,
                },
                {
                    "candidate": "bad_final_good_tune",
                    "wf_period": "final",
                    "selectable": True,
                    "status": "OK",
                    "bet_count": 30,
                    "roi": -0.50,
                    "mean_clv": 0.01,
                    "calibration_error": 0.03,
                },
                {
                    "candidate": "great_final_bad_tune",
                    "wf_period": "final",
                    "selectable": True,
                    "status": "OK",
                    "bet_count": 30,
                    "roi": 0.50,
                    "mean_clv": 0.05,
                    "calibration_error": 0.02,
                },
            ]
        )

        candidate, status = select_candidate_on_tune(experiments)

        self.assertEqual(candidate, "bad_final_good_tune")
        self.assertEqual(status, "TUNE_PASS")


if __name__ == "__main__":
    unittest.main()
