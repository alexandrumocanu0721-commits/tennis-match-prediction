from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for scripts_dir in [ROOT / "scripts" / "atp", ROOT / "scripts" / "challenger"]:
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

import challenger_config as cfg  # noqa: E402
from challenger_utils import (  # noqa: E402
    build_external_atp_rankings_context,
    initialize_challenger_player,
    refresh_challenger_player_rankings,
)
from tennis_pipeline import compute_winner_perspective_diffs  # noqa: E402


ATP_SCRIPTS = [
    "scripts/atp/tennis_pipeline.py",
    "scripts/atp/build_features.py",
    "scripts/atp/train_model.py",
    "scripts/atp/evaluate_model.py",
    "scripts/atp/predict_match.py",
    "scripts/atp/backtest_predictions.py",
    "scripts/atp/clv_calculator.py",
]

CHALLENGER_SCRIPTS = [
    "scripts/challenger/challenger_config.py",
    "scripts/challenger/challenger_utils.py",
    "scripts/challenger/build_challenger_features.py",
    "scripts/challenger/train_challenger_model.py",
    "scripts/challenger/backtest_challenger_predictions.py",
    "scripts/challenger/clv_challenger_calculator.py",
]

FORBIDDEN_ATP_PATH_REFERENCES = [
    "load_match_history_csvs",
    "path_raw_dir",
    "path_processed_features_csv",
    "path_trained_model_pkl",
    "data/raw/atp/atp_matches_",
    "data/processed/atp/features.csv",
    "models/atp/xgboost_model.pkl",
]

CHALLENGER_PROCESSED_OUTPUTS = [
    "path_challenger_processed_features_csv",
    "path_challenger_backtest_predictions_csv",
    "path_challenger_clv_results_csv",
    "path_challenger_clv_unmatched_csv",
]


class ChallengerSafetyTests(unittest.TestCase):
    def test_challenger_config_paths_are_isolated(self):
        root = cfg.project_root()
        self.assertEqual(
            cfg.path_challenger_raw_dir(root),
            root / "data" / "raw" / "challenger",
        )
        self.assertEqual(
            cfg.path_challenger_processed_features_csv(root),
            root / "data" / "processed" / "challenger" / "features.csv",
        )
        self.assertEqual(
            cfg.path_challenger_predict_today_csv(root),
            root / "data" / "predict" / "challenger" / "today_matches.csv",
        )
        self.assertEqual(
            cfg.path_challenger_model_pkl(root),
            root / "models" / "challenger" / "challenger_xgboost_model.pkl",
        )
        self.assertEqual(
            cfg.path_external_atp_rankings_csv(root),
            root / "data" / "raw" / "atp" / "atp_rankings_20s.csv",
        )

    def test_challenger_match_glob_only_points_to_raw_chal(self):
        self.assertEqual(cfg.CHALLENGER_MATCH_GLOB, "atp_matches_qual_chall_*.csv")
        raw_dir = cfg.path_challenger_raw_dir(ROOT)
        for csv_path in raw_dir.glob(cfg.CHALLENGER_MATCH_GLOB):
            self.assertTrue(str(csv_path.resolve()).startswith(str(raw_dir.resolve())))
            self.assertTrue(csv_path.name.startswith("atp_matches_qual_chall_"))

    def test_challenger_scripts_do_not_import_atp_path_helpers(self):
        for script in CHALLENGER_SCRIPTS:
            text = (ROOT / script).read_text(encoding="utf-8")
            for forbidden in FORBIDDEN_ATP_PATH_REFERENCES:
                self.assertNotIn(
                    forbidden,
                    text,
                    msg=f"Forbidden ATP reference '{forbidden}' found in {script}",
                )

    def test_challenger_feature_builder_does_not_write_atp_features(self):
        script = ROOT / "scripts" / "challenger" / "build_challenger_features.py"
        text = script.read_text(encoding="utf-8")
        self.assertIn("path_challenger_processed_features_csv", text)
        self.assertNotIn("data/processed/atp/features.csv", text)

    def test_challenger_training_does_not_write_atp_model(self):
        script = ROOT / "scripts" / "challenger" / "train_challenger_model.py"
        text = script.read_text(encoding="utf-8")
        self.assertIn("path_challenger_model_pkl", text)
        self.assertNotIn("path_trained_model_pkl", text)
        self.assertNotIn("models/atp/xgboost_model.pkl", text)

    def test_challenger_outputs_stay_under_processed_challenger(self):
        root = cfg.project_root()
        processed_chal = (root / "data" / "processed" / "challenger").resolve()
        processed_atp = (root / "data" / "processed" / "atp").resolve()

        for helper_name in CHALLENGER_PROCESSED_OUTPUTS:
            output_path = getattr(cfg, helper_name)(root).resolve()
            self.assertIn(processed_chal, output_path.parents)
            self.assertNotIn(processed_atp, output_path.parents)

    def test_challenger_model_path_is_isolated(self):
        root = cfg.project_root()
        self.assertEqual(
            cfg.path_challenger_model_pkl(root),
            root / "models" / "challenger" / "challenger_xgboost_model.pkl",
        )
        self.assertNotEqual(
            cfg.path_challenger_model_pkl(root),
            root / "models" / "atp" / "xgboost_model.pkl",
        )

    def test_atp_scripts_do_not_import_challenger_modules(self):
        for script in ATP_SCRIPTS:
            text = (ROOT / script).read_text(encoding="utf-8")
            self.assertNotIn("challenger_", text)
            self.assertNotIn("build_challenger_features", text)

    def test_external_rankings_allowed_but_atp_match_reads_forbidden(self):
        self.assertTrue(cfg.USE_EXTERNAL_ATP_RANKINGS_FOR_CHALLENGER)
        self.assertEqual(cfg.CHALLENGER_DEFAULT_PLAYER_RANK, 2000)
        self.assertEqual(cfg.CHALLENGER_DEFAULT_PLAYER_POINTS, 0)

        all_text = "\n".join(
            (ROOT / script).read_text(encoding="utf-8")
            for script in CHALLENGER_SCRIPTS
        )
        self.assertIn("atp_rankings_20s.csv", all_text)
        self.assertNotIn("data/raw/atp/atp_matches_", all_text)

    def test_challenger_rank_refresh_is_time_aware_and_momentum_is_12_week(self):
        rankings_df = pd.DataFrame(
            [
                {"ranking_date": "20251001", "rank": 100, "player": 1, "points": 100},
                {"ranking_date": "20260112", "rank": 90, "player": 1, "points": 150},
                {"ranking_date": "20260112", "rank": 120, "player": 2, "points": 80},
            ]
        )
        rankings_df["ranking_date"] = pd.to_datetime(
            rankings_df["ranking_date"], format="%Y%m%d"
        )
        rankings_by_player = build_external_atp_rankings_context(rankings_df)

        players: dict = {}
        early_date = pd.Timestamp("2025-10-15")
        late_date = pd.Timestamp("2026-01-15")

        initialize_challenger_player(
            players,
            1,
            "P1",
            early_date,
            rankings_by_player,
            use_external_rankings=True,
        )
        initialize_challenger_player(
            players,
            2,
            "P2",
            early_date,
            rankings_by_player,
            use_external_rankings=True,
        )

        refresh_challenger_player_rankings(
            players,
            1,
            2,
            early_date,
            rankings_by_player=rankings_by_player,
            use_external_rankings=True,
        )
        self.assertEqual(players[1]["rank"], 100)
        self.assertEqual(players[1]["points"], 100)
        self.assertEqual(players[2]["rank"], 2000)
        self.assertEqual(players[2]["points"], 0)

        refresh_challenger_player_rankings(
            players,
            1,
            2,
            late_date,
            rankings_by_player=rankings_by_player,
            use_external_rankings=True,
        )
        self.assertEqual(players[1]["rank"], 90)
        self.assertEqual(players[1]["points"], 150)
        self.assertEqual(players[2]["rank"], 120)
        self.assertEqual(players[2]["points"], 80)

        diffs = compute_winner_perspective_diffs(
            winner=1,
            loser=2,
            match_date=late_date,
            surface="Hard",
            players=players,
            rankings_by_player=rankings_by_player,
            h2h_diffs=None,
            fatigue_stats=None,
        )
        expected_p1_momentum = (150 - 100) / (100 + 1)
        expected_p2_momentum = 0.0
        self.assertEqual(diffs["rank_diff"], 30)
        self.assertEqual(diffs["points_diff"], 70)
        self.assertAlmostEqual(
            diffs["rank_momentum_diff"],
            expected_p1_momentum - expected_p2_momentum,
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
