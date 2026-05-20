# =============================================================================
# evaluate_model.py — Offline metrics + calibration + SHAP for saved model
# =============================================================================
# Reloads the trained XGBoost, applies the same temporal split as train_model
# (train strictly before MODEL_EVAL_CUTOFF_DATE, evaluate from that date on), then prints
# metrics and opens diagnostic plots. Paths use repo root via tennis_pipeline.
# =============================================================================

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, log_loss
from xgboost import plot_importance

from tennis_pipeline import (
    FEATURES,
    TARGET,
    path_processed_features_csv,
    path_trained_model_pkl,
    project_root,
    temporal_train_test_split_for_modeling,
)


def main() -> None:
    root = project_root()
    df = pd.read_csv(path_processed_features_csv(root))

    # Must match train_model: train date < cutoff, test date >= MODEL_EVAL_CUTOFF_DATE.
    train_df, test_df = temporal_train_test_split_for_modeling(df, date_column="date")

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    model = joblib.load(path_trained_model_pkl(root))

    # Metrics on the held-out year only (same rows Optuna used for log loss).
    preds = model.predict(X_test)
    pred_probs = model.predict_proba(X_test)[:, 1]
    print()
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Log Loss:", log_loss(y_test, pred_probs))

    # Calibration: predicted probability vs empirical win rate in bins.
    prob_true, prob_pred = calibration_curve(y_test, pred_probs, n_bins=10)
    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Win Rate")
    plt.title("Calibration Curve")
    plt.show()

    # XGBoost built-in importance (gain / frequency style).
    plt.figure(figsize=(10, 8))
    plot_importance(model)
    plt.title("XGBoost Feature Importance")
    plt.show()

    # SHAP: global behavior on test set + one local waterfall.
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)
    shap.plots.waterfall(shap_values[0])


if __name__ == "__main__":
    main()
