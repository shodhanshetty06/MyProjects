"""Unit tests for the pure helper functions in `app.modules.analytics.service`
— trend-bucket filling, country guessing, and skill counting. No DB required.
"""
from __future__ import annotations

from datetime import UTC, datetime

from app.modules.analytics.schemas import TrendInterval
from app.modules.analytics.service import (
    _bucket_starts,
    _count_skills,
    _fill_trend,
    _guess_country,
    _truncate,
)


def test_truncate_day_zeroes_time_components() -> None:
    dt = datetime(2026, 3, 5, 14, 32, 7, tzinfo=UTC)
    assert _truncate(dt, TrendInterval.DAY) == datetime(2026, 3, 5, tzinfo=UTC)


def test_truncate_week_rounds_down_to_monday() -> None:
    # 2026-03-05 is a Thursday.
    dt = datetime(2026, 3, 5, 14, 32, tzinfo=UTC)
    assert _truncate(dt, TrendInterval.WEEK) == datetime(2026, 3, 2, tzinfo=UTC)


def test_truncate_month_rounds_down_to_first_of_month() -> None:
    dt = datetime(2026, 3, 5, tzinfo=UTC)
    assert _truncate(dt, TrendInterval.MONTH) == datetime(2026, 3, 1, tzinfo=UTC)


def test_bucket_starts_spans_inclusive_range_by_day() -> None:
    date_from = datetime(2026, 3, 1, tzinfo=UTC)
    date_to = datetime(2026, 3, 3, tzinfo=UTC)
    buckets = _bucket_starts(date_from, date_to, TrendInterval.DAY)
    assert buckets == [
        datetime(2026, 3, 1, tzinfo=UTC),
        datetime(2026, 3, 2, tzinfo=UTC),
        datetime(2026, 3, 3, tzinfo=UTC),
    ]


def test_bucket_starts_crosses_year_boundary_by_month() -> None:
    date_from = datetime(2025, 11, 15, tzinfo=UTC)
    date_to = datetime(2026, 1, 15, tzinfo=UTC)
    buckets = _bucket_starts(date_from, date_to, TrendInterval.MONTH)
    assert buckets == [
        datetime(2025, 11, 1, tzinfo=UTC),
        datetime(2025, 12, 1, tzinfo=UTC),
        datetime(2026, 1, 1, tzinfo=UTC),
    ]


def test_fill_trend_zero_fills_missing_buckets() -> None:
    date_from = datetime(2026, 3, 1, tzinfo=UTC)
    date_to = datetime(2026, 3, 3, tzinfo=UTC)
    rows = [(datetime(2026, 3, 2, tzinfo=UTC), 5)]

    points = _fill_trend(rows, date_from=date_from, date_to=date_to, interval=TrendInterval.DAY)

    assert [p.count for p in points] == [0, 5, 0]


def test_guess_country_uses_last_comma_segment() -> None:
    assert _guess_country("San Francisco, CA, USA") == "USA"
    assert _guess_country("Berlin, Germany") == "Germany"


def test_guess_country_handles_no_comma() -> None:
    assert _guess_country("Remote") == "Remote"


def test_guess_country_defaults_to_unknown_for_blank_input() -> None:
    assert _guess_country(None) == "Unknown"
    assert _guess_country("   ") == "Unknown"


def test_count_skills_dedupes_within_a_resume_but_counts_across_resumes() -> None:
    skill_lists = [
        ["Python", "python", "SQL"],  # duplicate within one resume, case-insensitive
        ["Python", "Docker"],
    ]

    counter, display_names = _count_skills(skill_lists)

    assert counter["python"] == 2
    assert counter["sql"] == 1
    assert counter["docker"] == 1
    assert display_names["python"] == "Python"


def test_count_skills_ignores_blank_entries() -> None:
    counter, _ = _count_skills([["Python", "", "  "]])
    assert counter["python"] == 1
    assert len(counter) == 1
