from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ATP_INPUT = PROJECT_ROOT / "data" / "processed" / "atp" / "backtest_predictions.csv"
CHALLENGER_INPUT = PROJECT_ROOT / "data" / "processed" / "challenger" / "clv_results.csv"

EQUITY_OUTPUT = PROJECT_ROOT / "data" / "processed" / "strategy_backtest_equity.csv"
MONTHLY_OUTPUT = PROJECT_ROOT / "data" / "processed" / "strategy_backtest_monthly.csv"

STARTING_BANKROLL = 200.0
MIN_EDGE = 0.03
MAX_EDGE = 0.12
MAX_DAILY_EXPOSURE = 0.95
CUTOFF_DATE = pd.Timestamp("2025-01-01")

TOUR_SORT_ORDER = {"ATP": 0, "Challenger": 1}
SURFACE_ORDER = ["Hard", "Clay", "Grass"]
EDGE_TIER_ORDER = ["Small", "Medium", "Strong"]


def _exit_with_error(message: str) -> None:
    print(f"ERROR: {message}")
    raise SystemExit(1)


def _find_column(path: Path, columns: pd.Index, label: str, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    expected = ", ".join(candidates)
    _exit_with_error(f"missing required column for {label} in {path}: expected one of [{expected}]")
    raise AssertionError("unreachable")


def _load_source(
    path: Path,
    tour: str,
    column_map: dict[str, list[str]],
) -> pd.DataFrame:
    if not path.exists():
        _exit_with_error(f"missing input file: {path}")

    df = pd.read_csv(path)

    selected_columns: dict[str, str] = {}
    for target, candidates in column_map.items():
        selected_columns[target] = _find_column(path, df.columns, target, candidates)

    working = df.loc[:, list(selected_columns.values())].rename(
        columns={source: target for target, source in selected_columns.items()}
    )
    working["tour"] = tour
    working["source_file"] = str(path)
    working["source_row_idx"] = df.index
    working["date"] = pd.to_datetime(working["date"], errors="coerce")

    invalid_dates = working["date"].isna()
    if invalid_dates.any():
        for _, row in working.loc[invalid_dates, ["source_file", "source_row_idx"]].iterrows():
            print(
                f"WARNING: skipped row {int(row['source_row_idx'])} in {row['source_file']} "
                "because date is missing or invalid"
            )
        working = working.loc[~invalid_dates].copy()

    working = working.loc[working["date"].ge(CUTOFF_DATE)].copy()

    working["edge"] = pd.to_numeric(working["edge"], errors="coerce")
    working["odds"] = pd.to_numeric(working["odds"], errors="coerce")
    working["result"] = pd.to_numeric(working["result"], errors="coerce")
    working["surface"] = working["surface"].fillna("Unknown")
    working["player_a"] = working["player_a"].fillna("")
    working["player_b"] = working["player_b"].fillna("")

    return working


def _edge_tier(edge: float) -> str | None:
    if MIN_EDGE <= edge < 0.05:
        return "Small"
    if 0.05 <= edge < 0.08:
        return "Medium"
    if 0.08 <= edge <= MAX_EDGE:
        return "Strong"
    return None


def _stake_pct(edge: float) -> float:
    tier = _edge_tier(edge)
    if tier == "Small":
        return 0.02
    if tier == "Medium":
        return 0.03
    if tier == "Strong":
        return 0.05
    return 0.0


def _fmt_money(value: float) -> str:
    return f"{value:.2f}"


def _fmt_pct(value: float | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def _summarize_group(df: pd.DataFrame) -> dict[str, float | int]:
    bet_count = int(len(df))
    stakes = float(df["stake_eur"].sum())
    returns = float(df["returns_eur"].sum())
    pnl = float(df["pnl_eur"].sum())
    wins = int(df["result"].eq(1).sum())

    return {
        "bet_count": bet_count,
        "hit_rate": (wins / bet_count) if bet_count else float("nan"),
        "stakes": stakes,
        "returns": returns,
        "pnl": pnl,
        "roi": ((returns - stakes) / stakes) if stakes else float("nan"),
    }


def _floor_to_cents(series: pd.Series) -> pd.Series:
    return (series.mul(100).astype(int) / 100).astype(float)


def _print_row_summary(label: str, df: pd.DataFrame, include_pnl: bool = False) -> None:
    summary = _summarize_group(df)
    line = (
        f"  {label}: {summary['bet_count']} bets, "
        f"hit rate {_fmt_pct(summary['hit_rate'])}, ROI {_fmt_pct(summary['roi'])}"
    )
    if include_pnl:
        line += f", net P&L {_fmt_money(summary['pnl'])} EUR"
    print(line)


def _longest_losing_streak(df: pd.DataFrame) -> int:
    streak = 0
    best = 0
    for result in df["result"]:
        if result == 0:
            streak += 1
            best = max(best, streak)
        elif result == 1:
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


def _build_monthly_summary(equity_df: pd.DataFrame) -> pd.DataFrame:
    if equity_df.empty:
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
        equity_df.assign(month=equity_df["date"].dt.to_period("M").astype(str))
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
    atp = _load_source(
        ATP_INPUT,
        "ATP",
        {
            "date": ["date"],
            "edge": ["clv_b365"],
            "odds": ["odds_a_b365", "odds_a"],
            "result": ["actual_winner"],
            "player_a": ["player_a", "player_a_name"],
            "player_b": ["player_b", "player_b_name"],
            "surface": ["surface"],
        },
    )
    challenger = _load_source(
        CHALLENGER_INPUT,
        "Challenger",
        {
            "date": ["date"],
            "edge": ["clv_bet365"],
            "odds": ["odds_a"],
            "result": ["actual_result"],
            "player_a": ["player_a_name", "player_a"],
            "player_b": ["player_b_name", "player_b"],
            "surface": ["surface"],
        },
    )

    combined = pd.concat([atp, challenger], ignore_index=True)
    combined["tour_sort"] = combined["tour"].map(TOUR_SORT_ORDER).fillna(99).astype(int)
    combined = combined.sort_values(["date", "tour_sort", "source_row_idx"], kind="stable").reset_index(drop=True)

    required_mask = combined["edge"].notna() & combined["odds"].notna() & combined["result"].notna()
    missing_required = combined.loc[~required_mask]
    for _, row in missing_required.iterrows():
        missing = []
        if pd.isna(row["edge"]):
            missing.append("edge")
        if pd.isna(row["odds"]):
            missing.append("odds")
        if pd.isna(row["result"]):
            missing.append("result")
        print(
            f"WARNING: skipped row {int(row['source_row_idx'])} in {row['source_file']} "
            f"because {', '.join(missing)} is missing"
        )

    working = combined.loc[required_mask].copy()

    invalid_odds = working["odds"].le(1.0)
    for _, row in working.loc[invalid_odds].iterrows():
        print(
            f"WARNING: skipped row {int(row['source_row_idx'])} in {row['source_file']} "
            f"because odds is invalid ({row['odds']})"
        )
    working = working.loc[~invalid_odds].copy()

    invalid_result = ~working["result"].isin([0, 1])
    for _, row in working.loc[invalid_result].iterrows():
        print(
            f"WARNING: skipped row {int(row['source_row_idx'])} in {row['source_file']} "
            f"because result is invalid ({row['result']})"
        )
    working = working.loc[~invalid_result].copy()

    eligible = working.loc[working["edge"].between(MIN_EDGE, MAX_EDGE, inclusive="both")].copy()
    eligible["edge_tier"] = eligible["edge"].map(_edge_tier)
    eligible["stake_pct"] = eligible["edge"].map(_stake_pct)
    eligible = eligible.sort_values(["date", "tour_sort", "source_row_idx"], kind="stable").reset_index(drop=True)

    if eligible.empty:
        print("=== Strategy Backtest: ATP + Challenger Combined ===")
        print("Date range: N/A to N/A")
        print("Total bets: 0")
        print("  ATP bets: 0")
        print("  Challenger bets: 0")
        print()
        print(f"Starting bankroll: {_fmt_money(STARTING_BANKROLL)} EUR")
        print(f"Final bankroll: {_fmt_money(STARTING_BANKROLL)} EUR")
        print(f"Net P&L: {_fmt_money(0.0)} EUR")
        print(f"Overall ROI: {_fmt_pct(float('nan'))}")
        print()
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
        day["stake_eur"] = day_bankroll * day["stake_pct"]

        total_exposure = float(day["stake_eur"].sum())
        exposure_cap = day_bankroll * MAX_DAILY_EXPOSURE
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

        daily_bankroll_rows.append(
            {
                "date": date_value,
                "bankroll_after_day": bankroll,
            }
        )

        for _, row in day.iterrows():
            equity_rows.append(
                {
                    "date": row["date"],
                    "tour": row["tour"],
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
        equity = equity.sort_values(["date", "tour"], key=lambda s: s.map(TOUR_SORT_ORDER) if s.name == "tour" else s, kind="stable").reset_index(drop=True)
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

    total_stakes = float(equity["stake_eur"].sum())
    total_returns = float(equity["returns_eur"].sum())
    total_pnl = float(equity["pnl_eur"].sum())
    overall_roi = ((total_returns - total_stakes) / total_stakes) if total_stakes else float("nan")
    worst_dd_abs, worst_dd_pct = _worst_drawdown(daily_bankroll)
    longest_losing_streak = _longest_losing_streak(equity)

    print("=== Strategy Backtest: ATP + Challenger Combined ===")
    print(
        "Date range: "
        f"{equity['date'].min().strftime('%Y-%m-%d')} to {equity['date'].max().strftime('%Y-%m-%d')}"
    )
    print(f"Total bets: {len(equity)}")
    print(f"  ATP bets: {int(equity['tour'].eq('ATP').sum())}")
    print(f"  Challenger bets: {int(equity['tour'].eq('Challenger').sum())}")
    print()
    print(f"Starting bankroll: {_fmt_money(STARTING_BANKROLL)} EUR")
    print(f"Final bankroll: {_fmt_money(float(equity['bankroll_after'].iloc[-1]))} EUR")
    print(f"Net P&L: {_fmt_money(total_pnl)} EUR")
    print(f"Overall ROI: {_fmt_pct(overall_roi)}")
    if cap_trigger_dates:
        print(
            "Exposure cap triggered on "
            f"{len(cap_trigger_dates)} day(s); total same-day exposure was scaled to 95% of bankroll."
        )
    print()
    print("Bets by edge tier:")
    for tier_label, display_label in [
        ("Small", "Small  (3-5%)"),
        ("Medium", "Medium (5-8%)"),
        ("Strong", "Strong (8-12%)"),
    ]:
        _print_row_summary(display_label, equity.loc[equity["edge_tier"].eq(tier_label)])
    print()
    print("Bets by tour:")
    _print_row_summary("ATP       ", equity.loc[equity["tour"].eq("ATP")], include_pnl=True)
    _print_row_summary("Challenger", equity.loc[equity["tour"].eq("Challenger")], include_pnl=True)
    print()
    print("Bets by surface:")
    for surface in SURFACE_ORDER:
        _print_row_summary(surface.ljust(5), equity.loc[equity["surface"].eq(surface)])
    print()
    print("Bets by year:")
    equity["year"] = equity["date"].dt.year
    for year in sorted(equity["year"].dropna().astype(int).unique()):
        _print_row_summary(str(year), equity.loc[equity["year"].eq(year)], include_pnl=True)
    print()
    print(
        "Worst drawdown: "
        f"{_fmt_money(worst_dd_abs)} EUR ({worst_dd_pct * 100:.1f}% of peak bankroll)"
    )
    print(f"Longest losing streak: {longest_losing_streak} bets")


if __name__ == "__main__":
    main()
