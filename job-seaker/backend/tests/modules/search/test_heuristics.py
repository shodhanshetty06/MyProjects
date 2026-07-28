"""Unit tests for `search/heuristics.py` — pure, DB-free table-driven
boundary-value tests. No DB/Postgres connection required.
"""

from __future__ import annotations

import pytest

from app.modules.search.heuristics import (
    compute_application_difficulty,
    compute_hiring_probability,
    estimate_ats_score,
)

# --- compute_hiring_probability -----------------------------------------------------


@pytest.mark.parametrize(
    ("hiring_frequency_per_month", "days_since_posted", "expected"),
    [
        # Day 0, unknown cadence (neutral 50): 0.6*100 + 0.4*50 = 80
        (None, 0, 80.0),
        # At/after the decay floor, unknown cadence: 0.6*20 + 0.4*50 = 32
        (None, 45, 32.0),
        (None, 90, 32.0),
        # Halfway through decay (22.5 days of 45): recency = 100 - 0.5*80 = 60
        # 0.6*60 + 0.4*50 = 56
        (None, 22.5, 56.0),
        # Day 0, saturating cadence (>=4/month hires): 0.6*100 + 0.4*100 = 100 -> clamped to 95
        (4.0, 0, 95.0),
        (10.0, 0, 95.0),
        # Day 0, zero cadence: 0.6*100 + 0.4*0 = 60
        (0.0, 0, 60.0),
        # Far past decay floor with zero cadence: 0.6*20 + 0.4*0 = 12 -> clamped to floor 5
        (0.0, 200, 12.0),
    ],
)
def test_compute_hiring_probability_boundaries(
    hiring_frequency_per_month: float | None, days_since_posted: float, expected: float
) -> None:
    result = compute_hiring_probability(
        hiring_frequency_per_month=hiring_frequency_per_month,
        days_since_posted=days_since_posted,
        recency_decay_days=45,
    )
    assert result == pytest.approx(expected, abs=0.01)


def test_compute_hiring_probability_never_exceeds_bounds() -> None:
    high = compute_hiring_probability(
        hiring_frequency_per_month=1000.0, days_since_posted=0, recency_decay_days=45
    )
    low = compute_hiring_probability(
        hiring_frequency_per_month=0.0, days_since_posted=10_000, recency_decay_days=45
    )
    assert high <= 95.0
    assert low >= 5.0


def test_compute_hiring_probability_negative_days_treated_as_zero() -> None:
    result = compute_hiring_probability(
        hiring_frequency_per_month=None, days_since_posted=-5, recency_decay_days=45
    )
    assert result == pytest.approx(80.0, abs=0.01)


# --- compute_application_difficulty -------------------------------------------------


def test_difficulty_all_signals_absent_is_easy() -> None:
    label, score = compute_application_difficulty(
        required_skill_count=3,
        experience_level_title="Software Engineer",
        description_length=500,
        skill_count_threshold=8,
    )
    assert label == "easy"
    assert score == 0.0


def test_difficulty_seniority_keyword_alone_is_easy() -> None:
    # A flat 30pt seniority contribution alone stays under the 35pt "easy"
    # ceiling — difficulty needs more than one signal to cross into moderate.
    label, score = compute_application_difficulty(
        required_skill_count=3,
        experience_level_title="Senior Software Engineer",
        description_length=500,
        skill_count_threshold=8,
    )
    assert score == 30.0
    assert label == "easy"


def test_difficulty_high_skill_count_contributes() -> None:
    label, score = compute_application_difficulty(
        required_skill_count=16,  # threshold(8) + 8 overflow -> saturates at 35
        experience_level_title="Software Engineer",
        description_length=500,
        skill_count_threshold=8,
    )
    assert score == 35.0
    assert label == "moderate"


def test_difficulty_long_description_contributes() -> None:
    label, score = compute_application_difficulty(
        required_skill_count=3,
        experience_level_title="Software Engineer",
        description_length=3000,
        skill_count_threshold=8,
    )
    assert score == 35.0
    assert label == "moderate"


def test_difficulty_all_signals_combined_saturates_hard() -> None:
    label, score = compute_application_difficulty(
        required_skill_count=20,
        experience_level_title="Principal Engineer",
        description_length=5000,
        skill_count_threshold=8,
    )
    assert label == "hard"
    assert score == 100.0


@pytest.mark.parametrize(
    ("score_inputs", "expected_label"),
    [
        # boundary: score just under 35 -> easy; exactly on a bucket edge is
        # exercised indirectly since these are derived scores, not raw floats.
        (dict(required_skill_count=1, experience_level_title="x", description_length=0), "easy"),
        (
            dict(required_skill_count=1, experience_level_title="Senior x", description_length=0),
            "easy",
        ),
    ],
)
def test_difficulty_bucket_boundaries(score_inputs: dict, expected_label: str) -> None:
    label, _ = compute_application_difficulty(skill_count_threshold=8, **score_inputs)
    assert label == expected_label


# --- estimate_ats_score --------------------------------------------------------------


def test_ats_score_none_without_resume_baseline() -> None:
    assert (
        estimate_ats_score(
            resume_ats_baseline_score=None, missing_skill_count=2, total_skill_count=5
        )
        is None
    )


def test_ats_score_no_missing_skills_equals_baseline() -> None:
    result = estimate_ats_score(
        resume_ats_baseline_score=80.0, missing_skill_count=0, total_skill_count=5
    )
    assert result == 80.0


def test_ats_score_all_missing_halves_baseline() -> None:
    result = estimate_ats_score(
        resume_ats_baseline_score=80.0, missing_skill_count=5, total_skill_count=5
    )
    assert result == 40.0


def test_ats_score_partial_missing_scales_linearly() -> None:
    # missing_ratio = 2/4 = 0.5 -> baseline * (1 - 0.25) = baseline * 0.75
    result = estimate_ats_score(
        resume_ats_baseline_score=100.0, missing_skill_count=2, total_skill_count=4
    )
    assert result == 75.0


def test_ats_score_zero_total_skills_treated_as_no_gap() -> None:
    result = estimate_ats_score(
        resume_ats_baseline_score=60.0, missing_skill_count=0, total_skill_count=0
    )
    assert result == 60.0


def test_ats_score_clamped_to_zero_floor() -> None:
    # A baseline that's already negative-adjacent shouldn't happen in
    # practice, but the clamp must still hold at the floor.
    result = estimate_ats_score(
        resume_ats_baseline_score=0.0, missing_skill_count=5, total_skill_count=5
    )
    assert result == 0.0
