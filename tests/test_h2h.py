from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts" / "atp"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tennis_pipeline import (  # noqa: E402
    compute_h2h_diffs,
    compute_h2h_win_rate,
    initialize_h2h_records,
    update_h2h_records,
)


class H2HWinRateTests(unittest.TestCase):
    def test_empty_h2h_returns_neutral(self):
        h2h_records = initialize_h2h_records()

        rate = compute_h2h_win_rate(
            h2h_records,
            player_a=1,
            player_b=2,
            match_date=pd.Timestamp("2026-01-01"),
        )

        self.assertEqual(rate, 0.5)

    def test_single_meeting_below_threshold_returns_neutral(self):
        h2h_records = initialize_h2h_records()
        h2h_records[frozenset({1, 2})] = [
            (pd.Timestamp("2025-12-31"), "Hard", 1),
        ]

        rate = compute_h2h_win_rate(
            h2h_records,
            player_a=1,
            player_b=2,
            match_date=pd.Timestamp("2026-01-01"),
        )

        self.assertEqual(rate, 0.5)

    def test_correct_winner_gets_rate_above_half(self):
        h2h_records = initialize_h2h_records()
        h2h_records[frozenset({1, 2})] = [
            (pd.Timestamp("2024-01-01"), "Hard", 1),
            (pd.Timestamp("2024-06-01"), "Clay", 1),
            (pd.Timestamp("2024-09-01"), "Grass", 2),
        ]

        rate = compute_h2h_win_rate(
            h2h_records,
            player_a=1,
            player_b=2,
            match_date=pd.Timestamp("2026-01-01"),
        )

        self.assertGreater(rate, 0.5)

    def test_old_matches_get_downweighted_vs_recent_ones(self):
        match_date = pd.Timestamp("2026-01-01")
        h2h_records = initialize_h2h_records()
        h2h_records[frozenset({1, 2})] = [
            (pd.Timestamp("2025-12-15"), "Hard", 1),
            (pd.Timestamp("2025-12-10"), "Hard", 1),
            (pd.Timestamp("2020-01-01"), "Hard", 2),
        ]

        rate_recent_plus_old_loss = compute_h2h_win_rate(
            h2h_records,
            player_a=1,
            player_b=2,
            match_date=match_date,
        )

        h2h_records[frozenset({3, 4})] = [
            (pd.Timestamp("2025-12-15"), "Hard", 3),
            (pd.Timestamp("2025-12-10"), "Hard", 3),
            (pd.Timestamp("2025-12-01"), "Hard", 4),
        ]

        rate_recent_split = compute_h2h_win_rate(
            h2h_records,
            player_a=3,
            player_b=4,
            match_date=match_date,
        )

        self.assertGreater(rate_recent_plus_old_loss, rate_recent_split)

    def test_surface_filter_works_correctly(self):
        h2h_records = initialize_h2h_records()
        match_date = pd.Timestamp("2026-01-01")
        h2h_records[frozenset({1, 2})] = [
            (pd.Timestamp("2025-12-15"), "Hard", 1),
            (pd.Timestamp("2025-12-10"), "Clay", 2),
            (pd.Timestamp("2025-12-05"), "Hard", 1),
        ]

        overall_rate = compute_h2h_win_rate(
            h2h_records,
            player_a=1,
            player_b=2,
            match_date=match_date,
        )
        hard_rate = compute_h2h_win_rate(
            h2h_records,
            player_a=1,
            player_b=2,
            match_date=match_date,
            surface_filter="Hard",
        )
        clay_rate = compute_h2h_win_rate(
            h2h_records,
            player_a=1,
            player_b=2,
            match_date=match_date,
            surface_filter="Clay",
        )

        self.assertNotEqual(hard_rate, clay_rate)
        self.assertNotEqual(overall_rate, clay_rate)

    def test_h2h_win_rate_is_antisymmetric(self):
        h2h_records = initialize_h2h_records()
        h2h_records[frozenset({1, 2})] = [
            (pd.Timestamp("2025-12-15"), "Hard", 1),
            (pd.Timestamp("2024-12-15"), "Clay", 2),
            (pd.Timestamp("2023-12-15"), "Grass", 1),
        ]

        rate_ab = compute_h2h_win_rate(
            h2h_records,
            player_a=1,
            player_b=2,
            match_date=pd.Timestamp("2026-01-01"),
        )
        rate_ba = compute_h2h_win_rate(
            h2h_records,
            player_a=2,
            player_b=1,
            match_date=pd.Timestamp("2026-01-01"),
        )

        self.assertAlmostEqual(rate_ab + rate_ba, 1.0)

    def test_match_three_snapshot_excludes_current_match(self):
        h2h_records = initialize_h2h_records()

        match1_date = pd.Timestamp("2025-01-01")
        match2_date = pd.Timestamp("2025-02-01")
        match3_date = pd.Timestamp("2025-03-01")

        update_h2h_records(h2h_records, winner="A", loser="B", surface="Hard", match_date=match1_date)
        update_h2h_records(h2h_records, winner="B", loser="A", surface="Hard", match_date=match2_date)

        snapshot_diffs = compute_h2h_diffs(
            h2h_records,
            winner="A",
            loser="B",
            surface="Hard",
            match_date=match3_date,
        )

        self.assertEqual(snapshot_diffs["h2h_win_rate_diff"], 0.0)
        self.assertEqual(snapshot_diffs["h2h_surface_win_rate_diff"], 0.0)

        update_h2h_records(
            h2h_records,
            winner="A",
            loser="B",
            surface="Hard",
            match_date=match3_date,
        )

        post_update_diffs = compute_h2h_diffs(
            h2h_records,
            winner="A",
            loser="B",
            surface="Hard",
            match_date=match3_date + pd.Timedelta(days=1),
        )

        self.assertAlmostEqual(post_update_diffs["h2h_win_rate_diff"], 1 / 3)
        self.assertAlmostEqual(
            post_update_diffs["h2h_surface_win_rate_diff"], 1 / 3
        )


if __name__ == "__main__":
    unittest.main()
