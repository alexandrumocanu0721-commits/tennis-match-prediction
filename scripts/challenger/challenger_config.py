from __future__ import annotations

import sys
from pathlib import Path

CHALLENGER_MATCH_GLOB = "atp_matches_qual_chall_*.csv"
CHALLENGER_MATCH_HISTORY_START_YEAR = 2019
CHALLENGER_ALLOWED_TOURNEY_LEVEL = "C"
CHALLENGER_INVALID_OUTCOME_REGEX = r"RET|W/O|Walkover|DEF"

CHALLENGER_MODEL_EVAL_CUTOFF_DATE = "2025-01-01"

CHALLENGER_DEFAULT_PLAYER_RANK = 2000
CHALLENGER_DEFAULT_PLAYER_POINTS = 0

# External public ranking context is allowed for Challenger features; this does
# not import ATP match-derived state (Elo/form/H2H/serve-return) into Challenger.
USE_EXTERNAL_ATP_RANKINGS_FOR_CHALLENGER = True


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ATP_SCRIPT_DIR = project_root() / "scripts" / "atp"
if str(ATP_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(ATP_SCRIPT_DIR))


def path_challenger_raw_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "data" / "raw" / "challenger"


def path_challenger_processed_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "data" / "processed" / "challenger"


def path_challenger_predict_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "data" / "predict" / "challenger"


def path_challenger_processed_features_csv(root: Path | None = None) -> Path:
    return path_challenger_processed_dir(root) / "features.csv"


def path_challenger_backtest_predictions_csv(root: Path | None = None) -> Path:
    return path_challenger_processed_dir(root) / "backtest_predictions.csv"


def path_challenger_clv_results_csv(root: Path | None = None) -> Path:
    return path_challenger_processed_dir(root) / "clv_results.csv"


def path_challenger_clv_cumulative_png(root: Path | None = None) -> Path:
    return path_challenger_processed_dir(root) / "clv_cumulative.png"


def path_challenger_clv_unmatched_csv(root: Path | None = None) -> Path:
    return path_challenger_processed_dir(root) / "clv_unmatched.csv"


def path_challenger_predict_today_csv(root: Path | None = None) -> Path:
    return path_challenger_predict_dir(root) / "today_matches.csv"


def path_challenger_predict_output_csv(root: Path | None = None) -> Path:
    return path_challenger_predict_dir(root) / "predictions.csv"


def path_challenger_model_pkl(root: Path | None = None) -> Path:
    return (root or project_root()) / "models" / "challenger" / "challenger_xgboost_model.pkl"


def path_external_atp_rankings_csv(root: Path | None = None) -> Path:
    return (root or project_root()) / "data" / "raw" / "atp" / "atp_rankings_20s.csv"


def assert_challenger_processed_output_path(path: Path, root: Path | None = None) -> None:
    repo_root = root or project_root()
    resolved_path = path.resolve()
    processed_chal = path_challenger_processed_dir(repo_root).resolve()
    processed_atp = (repo_root / "data" / "processed" / "atp").resolve()

    if processed_chal not in resolved_path.parents:
        raise ValueError(f"Refusing to write outside Challenger processed dir: {path}")
    if processed_atp in resolved_path.parents:
        raise ValueError(f"Refusing to write ATP processed output: {path}")


def assert_challenger_model_path(path: Path, root: Path | None = None) -> None:
    repo_root = root or project_root()
    resolved_path = path.resolve()
    expected = path_challenger_model_pkl(repo_root).resolve()
    atp_model = (repo_root / "models" / "atp" / "xgboost_model.pkl").resolve()

    if resolved_path != expected:
        raise ValueError(f"Unexpected Challenger model path: {path}")
    if resolved_path == atp_model:
        raise ValueError(f"Refusing to overwrite ATP model: {path}")
