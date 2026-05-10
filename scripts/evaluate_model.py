import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, log_loss
from xgboost import plot_importance

FEATURES = [
    "elo_diff",
    "surface_elo_diff",
    "rank_diff",
    "points_diff",
    "recent_form_diff",
    "recent_surface_form_diff",
    "win_pct_diff",
    "matches_played_diff",
]
TARGET = "result"

def main():
    df = pd.read_csv("./data/processed/features.csv")

    train_df = df[df["date"] < "2025-01-01"]
    test_df = df[df["date"] >= "2025-01-01"]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    model = joblib.load("./models/xgboost_model.pkl")

    preds = model.predict(X_test)
    pred_probs = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, preds)
    loss = log_loss(y_test, pred_probs)

    print()
    print("Accuracy:", accuracy)
    print("Log Loss:", loss)

    prob_true, prob_pred = calibration_curve(y_test, pred_probs, n_bins=10)

    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Win Rate")
    plt.title("Calibration Curve")
    plt.show()

    plt.figure(figsize=(10, 8))
    plot_importance(model)
    plt.title("XGBoost Feature Importance")
    plt.show()

    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)

    shap.plots.waterfall(shap_values[0])


if __name__ == "__main__":
    main()