from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "challenger" / "clv_results_calibrated.csv"
EQUITY_OUTPUT = PROJECT_ROOT / "data" / "processed" / "strategy_backtest_focused_equity.csv"
MONTHLY_OUTPUT = PROJECT_ROOT / "data" / "processed" / "strategy_backtest_focused_monthly.csv"

STARTING_BANKROLL = 200.0
EDGE_MIN = 0.03
EDGE_MAX = 0.08
STAKE_PCT = 0.01
DAILY_EXPOSURE_CAP = 0.10
CUTOFF_DATE = pd.Timestamp("2025-01-01")
ALLOWED_SURFACES = {"Hard", "Clay"}
MAX_ODDS = 1.50


def _exit_with_error(message: str) -> None:
    print(f"ERROR: {message}")
    raise SystemExit(1)


def _fmt_money(value: float) -> str:
    return f"{value:.2f}"


def _fmt_pct(value: float | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def _load_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        _exit_with_error(f"missing input file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = {
        "date",
        "player_a_name",
        "player_b_name",
        "surface",
        "calibrated_clv",
        "odds_a",
        "actual_result",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        _exit_with_error(f"missing required column(s) in {INPUT_CSV}: {', '.join(missing)}")

    working = df.copy()
    working["source_row_idx"] = working.index
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working["edge"] = pd.to_numeric(working["calibrated_clv"], errors="coerce")
    working["odds"] = pd.to_numeric(working["odds_a"], errors="coerce")
    working["result"] = pd.to_numeric(working["actual_result"], errors="coerce")
    working["player_a"] = working["player_a_name"].fillna("")
    working["player_b"] = working["player_b_name"].fillna("")

    invalid_dates = working["date"].isna()
    for _, row in working.loc[invalid_dates].iterrows():
        print(f"WARNING: skipped row {int(row['source_row_idx'])} because date is missing or invalid")
    working = working.loc[~invalid_dates].copy()
    working = working.loc[working["date"].ge(CUTOFF_DATE)].copy()

    missing_core = working["edge"].isna() | working["odds"].isna() | working["result"].isna()
    for _, row in working.loc[missing_core].iterrows():
        missing_fields = []
        if pd.isna(row["edge"]):
            missing_fields.append("edge")
        if pd.isna(row["odds"]):
            missing_fields.append("odds")
        if pd.isna(row["result"]):
            missing_fields.append("result")
        print(
            f"WARNING: skipped row {int(row['source_row_idx'])} "
            f"because {', '.join(missing_fields)} is missing"
        )
    working = working.loc[~missing_core].copy()

    invalid_odds = working["odds"].le(1.0)
    for _, row in working.loc[invalid_odds].iterrows():
        print(f"WARNING: skipped row {int(row['source_row_idx'])} because odds is invalid ({row['odds']})")
    working = working.loc[~invalid_odds].copy()

    invalid_result = ~working["result"].isin([0, 1])
    for _, row in working.loc[invalid_result].iterrows():
        print(f"WARNING: skipped row {int(row['source_row_idx'])} because result is invalid ({row['result']})")
    working = working.loc[~invalid_result].copy()

    return working


def _floor_to_cents(series: pd.Series) -> pd.Series:
    return (series.mul(100).astype(int) / 100).astype(float)


def _edge_tier(edge: float) -> str:
    if edge < 0.05:
        return "Small"
    return "Medium"


def _longest_losing_streak(df: pd.DataFrame) -> int:
    streak = 0
    best = 0
    for result in df["result"]:
        if result == 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def _worst_drawdown(daily_bankroll: pd.DataFrame) -> tuple[float, float]:
    peak = STARTING_BANKROLL
    worst_abs = 0.0
    worst_pct = 0.0
    for bankroll in daily_bankroll["bankroll_after_day"]:
        peak = max(peak, bankroll)
        drawdown = peak - bankroll
        drawdown_pct = (drawdown / peak) if peak else 0.0
        if drawdown > worst_abs:
            worst_abs = drawdown
            worst_pct = drawdown_pct
    return worst_abs, worst_pct


def _build_monthly_summary(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "bet_count",
                "hit_rate",
                "stakes_eur",
                "returns_eur",
                "pnl_eur",
                "roi",
                "bankroll_end",
            ]
        )

    monthly = (
        equity.assign(month=equity["date"].dt.to_period("M").astype(str))
        .groupby("month", sort=True, as_index=False)
        .agg(
            bet_count=("result", "size"),
            wins=("result", "sum"),
            stakes_eur=("stake_eur", "sum"),
            returns_eur=("returns_eur", "sum"),
            pnl_eur=("pnl_eur", "sum"),
            bankroll_end=("bankroll_after", "last"),
        )
    )
    monthly["hit_rate"] = monthly["wins"] / monthly["bet_count"]
    monthly["roi"] = (monthly["returns_eur"] - monthly["stakes_eur"]) / monthly["stakes_eur"]
    monthly = monthly.drop(columns=["wins"])
    for col in ["hit_rate", "stakes_eur", "returns_eur", "pnl_eur", "roi", "bankroll_end"]:
        monthly[col] = monthly[col].round(2)
    return monthly[
        [
            "month",
            "bet_count",
            "hit_rate",
            "stakes_eur",
            "returns_eur",
            "pnl_eur",
            "roi",
            "bankroll_end",
        ]
    ]


def main() -> None:
    df = _load_data()

    eligible = df.loc[
        df["surface"].isin(ALLOWED_SURFACES)
        & df["odds"].lt(MAX_ODDS)
        & df["edge"].between(EDGE_MIN, EDGE_MAX, inclusive="both")
    ].copy()
    eligible = eligible.sort_values(["date", "source_row_idx"], kind="stable").reset_index(drop=True)
    eligible["edge_tier"] = eligible["edge"].map(_edge_tier)

    if eligible.empty:
        print("=== Focused Strategy Backtest: Challenger Favorites ===")
        print("No eligible bets found.")
        _build_monthly_summary(pd.DataFrame()).to_csv(MONTHLY_OUTPUT, index=False)
        pd.DataFrame(
            columns=[
                "date",
                "tour",
                "player_a",
                "player_b",
                "surface",
                "edge",
                "edge_tier",
                "stake_eur",
                "odds",
                "result",
                "pnl_eur",
                "bankroll_after",
            ]
        ).to_csv(EQUITY_OUTPUT, index=False)
        return

    bankroll = STARTING_BANKROLL
    cap_trigger_dates: list[str] = []
    equity_rows: list[dict[str, object]] = []
    daily_bankroll_rows: list[dict[str, object]] = []

    for date_value, day_df in eligible.groupby("date", sort=True):
        if bankroll < 0.01:
            break

        day_bankroll = bankroll
        day = day_df.copy()
        day["stake_eur"] = day_bankroll * STAKE_PCT

        total_exposure = float(day["stake_eur"].sum())
        exposure_cap = day_bankroll * DAILY_EXPOSURE_CAP
        if total_exposure > exposure_cap and total_exposure > 0:
            scale = exposure_cap / total_exposure
            day["stake_eur"] = day["stake_eur"] * scale
            cap_trigger_dates.append(date_value.strftime("%Y-%m-%d"))

        day["stake_eur"] = _floor_to_cents(day["stake_eur"])
        day = day.loc[day["stake_eur"].gt(0)].copy()
        if day.empty:
            break

        day["returns_eur"] = 0.0
        wins = day["result"].eq(1)
        day.loc[wins, "returns_eur"] = day.loc[wins, "stake_eur"] * day.loc[wins, "odds"]
        day["pnl_eur"] = day["returns_eur"] - day["stake_eur"]
        bankroll = max(0.0, day_bankroll + float(day["pnl_eur"].sum()))
        day["bankroll_after"] = bankroll

        daily_bankroll_rows.append({"date": date_value, "bankroll_after_day": bankroll})

        for _, row in day.iterrows():
            equity_rows.append(
                {
                    "date": row["date"],
                    "tour": "Challenger",
                    "player_a": row["player_a"],
                    "player_b": row["player_b"],
                    "surface": row["surface"],
                    "edge": float(row["edge"]),
                    "edge_tier": row["edge_tier"],
                    "stake_eur": float(row["stake_eur"]),
                    "odds": float(row["odds"]),
                    "result": int(row["result"]),
                    "returns_eur": float(row["returns_eur"]),
                    "pnl_eur": float(row["pnl_eur"]),
                    "bankroll_after": float(row["bankroll_after"]),
                }
            )

    equity = pd.DataFrame(equity_rows)
    daily_bankroll = pd.DataFrame(daily_bankroll_rows)

    if not equity.empty:
        equity["date"] = pd.to_datetime(equity["date"], errors="coerce")

    monthly = _build_monthly_summary(equity)

    equity_output = equity[
        [
            "date",
            "tour",
            "player_a",
            "player_b",
            "surface",
            "edge",
            "edge_tier",
            "stake_eur",
            "odds",
            "result",
            "pnl_eur",
            "bankroll_after",
        ]
    ].copy()
    equity_output["date"] = equity_output["date"].dt.strftime("%Y-%m-%d")
    for col in ["edge", "stake_eur", "odds", "pnl_eur", "bankroll_after"]:
        decimals = 4 if col == "edge" else 2
        equity_output[col] = equity_output[col].round(decimals)
    equity_output.to_csv(EQUITY_OUTPUT, index=False)
    monthly.to_csv(MONTHLY_OUTPUT, index=False)

    total_bets = len(equity)
    total_stakes = float(equity["stake_eur"].sum())
    total_returns = float(equity["returns_eur"].sum())
    total_pnl = float(equity["pnl_eur"].sum())
    hit_rate = float(equity["result"].mean()) if total_bets else float("nan")
    roi = ((total_returns - total_stakes) / total_stakes) if total_stakes else float("nan")
    worst_dd_abs, worst_dd_pct = _worst_drawdown(daily_bankroll)
    longest_losing_streak = _longest_losing_streak(equity)

    print("=== Focused Strategy Backtest: Challenger Favorites ===")
    print(
        "Rules: Challenger only, Hard/Clay only, odds < 1.50, "
        "calibrated edge 0.03-0.08, flat 1.00% stake, 10.00% daily cap"
    )
    print(
        "Date range: "
        f"{equity['date'].min().strftime('%Y-%m-%d')} to {equity['date'].max().strftime('%Y-%m-%d')}"
    )
    print(f"Total bets: {total_bets}")
    print(f"  Hard bets: {int(equity['surface'].eq('Hard').sum())}")
    print(f"  Clay bets: {int(equity['surface'].eq('Clay').sum())}")
    print()
    print(f"Starting bankroll: {_fmt_money(STARTING_BANKROLL)} EUR")
    print(f"Final bankroll: {_fmt_money(float(equity['bankroll_after'].iloc[-1]))} EUR")
    print(f"Net P&L: {_fmt_money(total_pnl)} EUR")
    print(f"Hit rate: {_fmt_pct(hit_rate)}")
    print(f"Overall ROI: {_fmt_pct(roi)}")
    if cap_trigger_dates:
        print(f"Exposure cap triggered on {len(cap_trigger_dates)} day(s).")
    print()
    print("By edge tier:")
    for tier in ["Small", "Medium"]:
        tier_df = equity.loc[equity["edge_tier"].eq(tier)]
        tier_stakes = float(tier_df["stake_eur"].sum())
        tier_returns = float(tier_df["returns_eur"].sum())
        tier_roi = ((tier_returns - tier_stakes) / tier_stakes) if tier_stakes else float("nan")
        print(
            f"  {tier}: {len(tier_df)} bets, hit rate {_fmt_pct(float(tier_df['result'].mean()) if len(tier_df) else float('nan'))}, "
            f"ROI {_fmt_pct(tier_roi)}"
        )
    print()
    print("By surface:")
    for surface in ["Hard", "Clay"]:
        surface_df = equity.loc[equity["surface"].eq(surface)]
        surface_stakes = float(surface_df["stake_eur"].sum())
        surface_returns = float(surface_df["returns_eur"].sum())
        surface_roi = ((surface_returns - surface_stakes) / surface_stakes) if surface_stakes else float("nan")
        print(
            f"  {surface}: {len(surface_df)} bets, hit rate {_fmt_pct(float(surface_df['result'].mean()) if len(surface_df) else float('nan'))}, "
            f"ROI {_fmt_pct(surface_roi)}"
        )
    print()
    print(
        "Worst drawdown: "
        f"{_fmt_money(worst_dd_abs)} EUR ({worst_dd_pct * 100:.1f}% of peak bankroll)"
    )
    print(f"Longest losing streak: {longest_losing_streak} bets")


if __name__ == "__main__":
    main()
