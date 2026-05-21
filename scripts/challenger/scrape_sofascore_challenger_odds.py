from __future__ import annotations

import argparse
import json
import logging
import random
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import requests
except ModuleNotFoundError:
    requests = None  # type: ignore[assignment]


EVENTS_URL = "https://www.sofascore.com/api/v1/category/72/scheduled-events/{date}"
ODDS_URL = "https://www.sofascore.com/api/v1/sport/tennis/odds/1/{date}"

SOURCE_BOOKMAKER = "bet365_via_sofascore"
ODDS_SOURCE_CONFIDENCE = "latest_pre_match_assumed"

REPORT_COLUMNS = [
    "total event rows fetched",
    "valid challenger singles matches",
    "valid matches with odds",
    "missing odds excluded",
    "canceled excluded",
    "doubles excluded",
    "unfinished excluded",
    "duplicates removed",
    "rows written",
]

OUTPUT_COLUMNS = [
    "endpoint_date",
    "match_start_utc",
    "match_start_date_utc",
    "event_id",
    "tournament_name",
    "unique_tournament_name",
    "unique_tournament_id",
    "round_name",
    "round_slug",
    "surface",
    "home_player",
    "away_player",
    "home_player_id",
    "away_player_id",
    "home_country",
    "away_country",
    "home_rank",
    "away_rank",
    "winner_code",
    "home_score_sets",
    "away_score_sets",
    "home_initial_fractional",
    "away_initial_fractional",
    "home_latest_fractional",
    "away_latest_fractional",
    "home_open_decimal",
    "away_open_decimal",
    "home_latest_decimal",
    "away_latest_decimal",
    "home_latest_no_vig",
    "away_latest_no_vig",
    "source_bookmaker",
    "odds_source_confidence",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape historical ATP Challenger men's singles match odds from "
            "SofaScore with local JSON caching."
        )
    )
    parser.add_argument("--start-date", required=True, help="First endpoint date, YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Last endpoint date, YYYY-MM-DD.")
    parser.add_argument(
        "--output",
        default="sofascore_challenger_2026.csv",
        help="Output CSV path. Relative paths are resolved under data/sofascore/challenger/.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=3.0,
        help="Base polite delay after live requests. A random 0-2 second jitter is added.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached JSON files and re-request SofaScore endpoints.",
    )
    parser.add_argument(
        "--sample-events",
        dest="sample_event_id",
        default=None,
        help="Print joined verification details for one SofaScore event id.",
    )
    return parser.parse_args()


def setup_logging(sofascore_dir: Path) -> None:
    logging.basicConfig(
        filename=sofascore_dir / "scrape_errors.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_dates(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("--end-date must be on or after --start-date")

    days = (end - start).days
    return [start + timedelta(days=offset) for offset in range(days + 1)]


def build_session() -> requests.Session:
    if requests is None:
        raise SystemExit(
            "Missing dependency: requests. Install project requirements before scraping."
        )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.sofascore.com/tennis/",
        }
    )
    return session


def load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Failed to load cached JSON %s: %s", path, exc)
        return None


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def polite_sleep(sleep_seconds: float) -> None:
    if sleep_seconds <= 0:
        return
    time.sleep(random.uniform(sleep_seconds, sleep_seconds + 2.0))


def fetch_json(
    session: requests.Session,
    url: str,
    cache_path: Path,
    force_refresh: bool,
    endpoint_date: str,
    label: str,
    sleep_seconds: float,
    max_retries: int = 4,
) -> dict[str, Any] | list[Any] | None:
    if cache_path.exists() and not force_refresh:
        cached_payload = load_json(cache_path)
        if cached_payload is not None:
            return cached_payload

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, timeout=30)
        except requests.RequestException as exc:
            logging.warning(
                "%s request exception for %s attempt %s/%s: %s",
                label,
                endpoint_date,
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                time.sleep(min(2**attempt, 30) + random.uniform(0, 2))
            continue

        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError as exc:
                logging.warning(
                    "%s invalid JSON for %s attempt %s/%s: %s",
                    label,
                    endpoint_date,
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    time.sleep(min(2**attempt, 30) + random.uniform(0, 2))
                continue

            save_json(cache_path, payload)
            polite_sleep(sleep_seconds)
            return payload

        logging.warning(
            "%s HTTP %s for %s attempt %s/%s",
            label,
            response.status_code,
            endpoint_date,
            attempt,
            max_retries,
        )
        if attempt >= max_retries:
            break

        if response.status_code in {403, 429, 503}:
            time.sleep(random.uniform(60, 180))
        else:
            time.sleep(min(2**attempt, 30) + random.uniform(0, 2))

    logging.error("%s failed for endpoint date %s after %s attempts", label, endpoint_date, max_retries)
    return None


def write_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: pandas. Install project requirements before writing CSV output."
        ) from exc

    df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


def write_report(report: dict[str, int], report_path: Path) -> None:
    lines = [f"{key}: {report.get(key, 0)}" for key in REPORT_COLUMNS]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"duplicates removed: {report.get('duplicates removed', 0)}")
    print(f"Wrote scrape report to {report_path}")


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def contains_token(value: Any, token: str, case_sensitive: bool = True) -> bool:
    if value is None:
        return False

    if isinstance(value, list):
        items = [str(item) for item in value]
    elif isinstance(value, tuple | set):
        items = [str(item) for item in value]
    else:
        items = [str(value)]

    if case_sensitive:
        return any(token in item for item in items)

    token_lower = token.lower()
    return any(token_lower in item.lower() for item in items)


def extract_events(payload: dict[str, Any] | list[Any] | None) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [event for event in payload if isinstance(event, dict)]

    root = as_dict(payload)
    events = root.get("events")
    if isinstance(events, list):
        return [event for event in events if isinstance(event, dict)]

    scheduled = root.get("scheduledEvents")
    if isinstance(scheduled, list):
        return [event for event in scheduled if isinstance(event, dict)]

    return []


def start_timestamp_to_utc(start_timestamp: Any) -> tuple[str | None, str | None]:
    if start_timestamp is None:
        return None, None

    try:
        timestamp = float(start_timestamp)
    except (TypeError, ValueError):
        return None, None

    match_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return match_dt.isoformat().replace("+00:00", "Z"), match_dt.date().isoformat()


def is_canceled_event(event: dict[str, Any]) -> bool:
    status = as_dict(event.get("status"))
    status_type = str(status.get("type", "")).lower()
    return status_type in {"canceled", "cancelled", "postponed", "interrupted"}


def is_finished_result(event: dict[str, Any]) -> bool:
    status = as_dict(event.get("status"))
    return status.get("type") == "finished" and status.get("code") == 100


def is_challenger_event(event: dict[str, Any]) -> bool:
    tournament = as_dict(event.get("tournament"))
    category = as_dict(tournament.get("category"))
    return category.get("name") == "Challenger"


def is_mens_singles_event(event: dict[str, Any]) -> bool:
    filters = as_dict(event.get("eventFilters"))
    # 2026+ payloads: prefer explicit eventFilters when category or gender is present.
    if filters.get("category") is not None or filters.get("gender") is not None:
        return contains_token(
            filters.get("category"),
            "singles",
            case_sensitive=False,
        ) and contains_token(filters.get("gender"), "M", case_sensitive=True)

    # 2025 fallback: infer from season name and require both competitors to be men.
    season = as_dict(event.get("season"))
    season_name = str(season.get("name", "")).lower()
    if "singles" not in season_name or "men" not in season_name:
        return False

    home_team = as_dict(event.get("homeTeam"))
    away_team = as_dict(event.get("awayTeam"))
    return home_team.get("gender") == "M" and away_team.get("gender") == "M"


def has_scores(event: dict[str, Any]) -> bool:
    return isinstance(event.get("homeScore"), dict) and isinstance(event.get("awayScore"), dict)


def walk_markets(value: Any) -> list[dict[str, Any]]:
    markets: list[dict[str, Any]] = []

    if isinstance(value, dict):
        if isinstance(value.get("choices"), list) or isinstance(value.get("outcomes"), list):
            markets.append(value)

        for child in value.values():
            markets.extend(walk_markets(child))
    elif isinstance(value, list):
        for child in value:
            markets.extend(walk_markets(child))

    return markets


def odds_payload_for_event(
    odds_payload: dict[str, Any] | list[Any] | None, event_id: Any
) -> Any:
    if event_id is None:
        return None

    event_key = str(event_id)
    root = odds_payload
    if isinstance(root, dict):
        direct = root.get(event_key)
        if direct is not None:
            return direct

        for wrapper_key in ("odds", "markets", "events"):
            wrapper = root.get(wrapper_key)
            if isinstance(wrapper, dict):
                direct = wrapper.get(event_key)
                if direct is not None:
                    return direct
            elif isinstance(wrapper, list):
                for item in wrapper:
                    item_dict = as_dict(item)
                    if str(item_dict.get("id")) == event_key or str(item_dict.get("eventId")) == event_key:
                        return item_dict

    if isinstance(root, list):
        for item in root:
            item_dict = as_dict(item)
            if str(item_dict.get("id")) == event_key or str(item_dict.get("eventId")) == event_key:
                return item_dict

    return None


def choice_name(choice: dict[str, Any]) -> str:
    return str(first_non_empty(choice.get("name"), choice.get("sourceId"), choice.get("id"), "")).strip()


def is_false(value: Any) -> bool:
    if value is False:
        return True
    if isinstance(value, str):
        return value.lower() == "false"
    return False


def select_match_winner_market(event_odds_payload: Any) -> dict[str, Any] | None:
    for market in walk_markets(event_odds_payload):
        choices = market.get("choices")
        if not isinstance(choices, list):
            choices = market.get("outcomes")

        if (
            market.get("marketName") != "Full time"
            or market.get("marketPeriod") != "Match"
            or market.get("marketGroup") != "Home/Away"
            or not is_false(market.get("isLive"))
            or not isinstance(choices, list)
            or len(choices) != 2
        ):
            continue

        choice_dicts = [as_dict(choice) for choice in choices]
        names = {choice_name(choice) for choice in choice_dicts}
        if names == {"1", "2"}:
            market_copy = dict(market)
            market_copy["choices"] = choice_dicts
            return market_copy

    return None


def fraction_to_decimal(value: Any) -> float | None:
    if value is None:
        return None

    parts = str(value).strip().split("/")
    if len(parts) != 2:
        return None

    try:
        numerator = float(parts[0])
        denominator = float(parts[1])
    except ValueError:
        return None

    if denominator == 0:
        return None

    return numerator / denominator + 1.0


def no_vig_probabilities(o1: float | None, o2: float | None) -> tuple[float | None, float | None]:
    if o1 is None or o2 is None or o1 <= 0 or o2 <= 0:
        return None, None

    p1_raw = 1.0 / o1
    p2_raw = 1.0 / o2
    total = p1_raw + p2_raw
    if total <= 0:
        return None, None

    return p1_raw / total, p2_raw / total


def winner_name(event: dict[str, Any]) -> Any:
    winner_code = event.get("winnerCode")
    if winner_code == 1:
        return as_dict(event.get("homeTeam")).get("name")
    if winner_code == 2:
        return as_dict(event.get("awayTeam")).get("name")
    return winner_code


def choice_fractional(choice: dict[str, Any], latest: bool) -> Any:
    if latest:
        return first_non_empty(
            choice.get("fractionalValue"),
            choice.get("latestFractionalValue"),
            choice.get("currentFractionalValue"),
        )

    return first_non_empty(
        choice.get("initialFractionalValue"),
        choice.get("openingFractionalValue"),
        choice.get("openFractionalValue"),
    )


def print_sample_event(
    event: dict[str, Any],
    odds_payload: dict[str, Any] | list[Any] | None,
) -> None:
    event_id = event.get("id")
    home_player = as_dict(event.get("homeTeam"))
    away_player = as_dict(event.get("awayTeam"))
    tournament = as_dict(event.get("tournament"))
    match_start_utc, _ = start_timestamp_to_utc(event.get("startTimestamp"))

    home_latest_fractional = None
    away_latest_fractional = None
    home_latest_decimal = None
    away_latest_decimal = None
    home_latest_no_vig = None
    away_latest_no_vig = None

    event_odds = odds_payload_for_event(odds_payload, event_id)
    market = select_match_winner_market(event_odds) if event_odds is not None else None
    if market is not None:
        choices_by_name = {
            choice_name(choice): choice for choice in as_list(market.get("choices"))
        }
        home_choice = as_dict(choices_by_name.get("1"))
        away_choice = as_dict(choices_by_name.get("2"))
        home_latest_fractional = choice_fractional(home_choice, latest=True)
        away_latest_fractional = choice_fractional(away_choice, latest=True)
        home_latest_decimal = fraction_to_decimal(home_latest_fractional)
        away_latest_decimal = fraction_to_decimal(away_latest_fractional)
        home_latest_no_vig, away_latest_no_vig = no_vig_probabilities(
            home_latest_decimal,
            away_latest_decimal,
        )

    print("event_id:", event_id)
    print("player1 vs player2:", f"{home_player.get('name')} vs {away_player.get('name')}")
    print("tournament:", tournament.get("name"))
    print("surface:", extract_surface(event))
    print("match_start_utc:", match_start_utc)
    print("winner:", winner_name(event))
    print("home latest odds:", home_latest_decimal)
    print("away latest odds:", away_latest_decimal)
    print("home no-vig:", home_latest_no_vig)
    print("away no-vig:", away_latest_no_vig)


def score_sets(score: dict[str, Any]) -> str | None:
    set_scores = []
    for period in range(1, 6):
        value = score.get(f"period{period}")
        if value is not None:
            set_scores.append(str(value))

    if set_scores:
        return ",".join(set_scores)

    current = score.get("current")
    return str(current) if current is not None else None


def player_country(player: dict[str, Any]) -> Any:
    country = as_dict(player.get("country"))
    return first_non_empty(country.get("name"), country.get("alpha2"), country.get("alpha3"))


def player_rank(player: dict[str, Any]) -> Any:
    return first_non_empty(
        player.get("ranking"),
        player.get("rank"),
        player.get("currentRanking"),
        player.get("seed"),
    )


def extract_surface(event: dict[str, Any]) -> Any:
    tournament = as_dict(event.get("tournament"))
    unique_tournament = as_dict(tournament.get("uniqueTournament"))
    return first_non_empty(
        event.get("groundType"),
        event.get("surface"),
        tournament.get("groundType"),
        tournament.get("surface"),
        unique_tournament.get("groundType"),
        unique_tournament.get("surface"),
    )


def build_record(
    endpoint_date: str,
    event: dict[str, Any],
    market: dict[str, Any],
) -> dict[str, Any] | None:
    choices_by_name = {choice_name(choice): choice for choice in as_list(market.get("choices"))}
    home_choice = as_dict(choices_by_name.get("1"))
    away_choice = as_dict(choices_by_name.get("2"))

    home_initial_fractional = choice_fractional(home_choice, latest=False)
    away_initial_fractional = choice_fractional(away_choice, latest=False)
    home_latest_fractional = choice_fractional(home_choice, latest=True)
    away_latest_fractional = choice_fractional(away_choice, latest=True)

    home_open_decimal = fraction_to_decimal(home_initial_fractional)
    away_open_decimal = fraction_to_decimal(away_initial_fractional)
    home_latest_decimal = fraction_to_decimal(home_latest_fractional)
    away_latest_decimal = fraction_to_decimal(away_latest_fractional)
    home_latest_no_vig, away_latest_no_vig = no_vig_probabilities(
        home_latest_decimal,
        away_latest_decimal,
    )

    if home_latest_decimal is None or away_latest_decimal is None:
        return None

    match_start_utc, match_start_date_utc = start_timestamp_to_utc(event.get("startTimestamp"))
    tournament = as_dict(event.get("tournament"))
    unique_tournament = as_dict(tournament.get("uniqueTournament"))
    round_info = as_dict(event.get("roundInfo"))
    home_player = as_dict(event.get("homeTeam"))
    away_player = as_dict(event.get("awayTeam"))
    home_score = as_dict(event.get("homeScore"))
    away_score = as_dict(event.get("awayScore"))

    return {
        "endpoint_date": endpoint_date,
        "match_start_utc": match_start_utc,
        "match_start_date_utc": match_start_date_utc,
        "event_id": event.get("id"),
        "tournament_name": tournament.get("name"),
        "unique_tournament_name": unique_tournament.get("name"),
        "unique_tournament_id": unique_tournament.get("id"),
        "round_name": first_non_empty(round_info.get("name"), round_info.get("round")),
        "round_slug": round_info.get("slug"),
        "surface": extract_surface(event),
        "home_player": home_player.get("name"),
        "away_player": away_player.get("name"),
        "home_player_id": home_player.get("id"),
        "away_player_id": away_player.get("id"),
        "home_country": player_country(home_player),
        "away_country": player_country(away_player),
        "home_rank": player_rank(home_player),
        "away_rank": player_rank(away_player),
        "winner_code": event.get("winnerCode"),
        "home_score_sets": score_sets(home_score),
        "away_score_sets": score_sets(away_score),
        "home_initial_fractional": home_initial_fractional,
        "away_initial_fractional": away_initial_fractional,
        "home_latest_fractional": home_latest_fractional,
        "away_latest_fractional": away_latest_fractional,
        "home_open_decimal": home_open_decimal,
        "away_open_decimal": away_open_decimal,
        "home_latest_decimal": home_latest_decimal,
        "away_latest_decimal": away_latest_decimal,
        "home_latest_no_vig": home_latest_no_vig,
        "away_latest_no_vig": away_latest_no_vig,
        "source_bookmaker": SOURCE_BOOKMAKER,
        "odds_source_confidence": ODDS_SOURCE_CONFIDENCE,
    }


def scrape_date(
    session: requests.Session,
    endpoint_day: date,
    events_cache_dir: Path,
    odds_cache_dir: Path,
    force_refresh: bool,
    sleep_seconds: float,
    sample_event_id: str | None,
    report: dict[str, int],
) -> list[dict[str, Any]]:
    endpoint_date = endpoint_day.isoformat()
    events_payload = fetch_json(
        session=session,
        url=EVENTS_URL.format(date=endpoint_date),
        cache_path=events_cache_dir / f"{endpoint_date}.json",
        force_refresh=force_refresh,
        endpoint_date=endpoint_date,
        label="events",
        sleep_seconds=sleep_seconds,
    )
    odds_payload = fetch_json(
        session=session,
        url=ODDS_URL.format(date=endpoint_date),
        cache_path=odds_cache_dir / f"{endpoint_date}.json",
        force_refresh=force_refresh,
        endpoint_date=endpoint_date,
        label="odds",
        sleep_seconds=sleep_seconds,
    )

    if events_payload is None:
        logging.error("Skipping endpoint date %s because events payload failed", endpoint_date)
        return []

    events = extract_events(events_payload)
    report["total event rows fetched"] += len(events)

    if odds_payload is None:
        logging.error("Skipping endpoint date %s because odds payload failed", endpoint_date)
        return []

    records: list[dict[str, Any]] = []
    for event in events:
        if sample_event_id is not None and str(event.get("id")) == sample_event_id:
            print_sample_event(event, odds_payload)

        if is_canceled_event(event):
            report["canceled excluded"] += 1
            continue

        if not is_finished_result(event):
            report["unfinished excluded"] += 1
            continue

        if not is_challenger_event(event):
            continue

        if not is_mens_singles_event(event):
            report["doubles excluded"] += 1
            continue

        if not has_scores(event):
            continue

        report["valid challenger singles matches"] += 1

        event_id = event.get("id")
        event_odds = odds_payload_for_event(odds_payload, event_id)
        if event_odds is None:
            report["missing odds excluded"] += 1
            continue

        market = select_match_winner_market(event_odds)
        if market is None:
            report["missing odds excluded"] += 1
            continue

        record = build_record(endpoint_date, event, market)
        if record is not None:
            records.append(record)
            report["valid matches with odds"] += 1
        else:
            report["missing odds excluded"] += 1

    return records


def deduplicate_latest_endpoint(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    latest_by_event_id: dict[Any, dict[str, Any]] = {}
    without_event_id: list[dict[str, Any]] = []

    for record in records:
        event_id = record.get("event_id")
        if event_id is None:
            without_event_id.append(record)
            continue

        previous = latest_by_event_id.get(event_id)
        if previous is None or str(record.get("endpoint_date", "")) >= str(
            previous.get("endpoint_date", "")
        ):
            latest_by_event_id[event_id] = record

    deduped = list(latest_by_event_id.values()) + without_event_id
    deduped.sort(
        key=lambda row: (
            str(row.get("match_start_utc") or ""),
            str(row.get("endpoint_date") or ""),
            str(row.get("event_id") or ""),
        )
    )
    duplicates_removed = len(records) - len(deduped)
    return deduped, duplicates_removed


def main() -> None:
    args = parse_args()
    root = project_root()
    sofascore_dir = root / "data" / "sofascore"
    challenger_sofascore_dir = sofascore_dir / "challenger"
    sofascore_dir.mkdir(parents=True, exist_ok=True)
    challenger_sofascore_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(challenger_sofascore_dir)

    start = parse_iso_date(args.start_date)
    end = parse_iso_date(args.end_date)

    events_cache_dir = sofascore_dir / "events"
    odds_cache_dir = sofascore_dir / "odds"
    events_cache_dir.mkdir(parents=True, exist_ok=True)
    odds_cache_dir.mkdir(parents=True, exist_ok=True)

    session = build_session()
    raw_valid_records: list[dict[str, Any]] = []
    report = {key: 0 for key in REPORT_COLUMNS}

    for endpoint_day in iter_dates(start, end):
        endpoint_records = scrape_date(
            session=session,
            endpoint_day=endpoint_day,
            events_cache_dir=events_cache_dir,
            odds_cache_dir=odds_cache_dir,
            force_refresh=args.force_refresh,
            sleep_seconds=args.sleep_seconds,
            sample_event_id=args.sample_event_id,
            report=report,
        )

        raw_valid_records.extend(endpoint_records)

        print(
            f"{endpoint_day.isoformat()}: kept {len(endpoint_records)} matches "
            f"({len(raw_valid_records)} valid rows before dedupe)"
        )

    all_records, duplicates_removed = deduplicate_latest_endpoint(raw_valid_records)
    report["duplicates removed"] = duplicates_removed
    report["rows written"] = len(all_records)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = challenger_sofascore_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_csv(all_records, output_path)
    write_report(report, challenger_sofascore_dir / "sofascore_challenger_scrape_report.txt")


if __name__ == "__main__":
    main()
