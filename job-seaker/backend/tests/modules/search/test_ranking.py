"""Unit tests for `search/ranking.py` — pure, DB-free table-driven tests,
same convention as `test_heuristics.py`.
"""

from __future__ import annotations

import pytest

from app.modules.search.ranking import (
    RankingWeights,
    compute_company_quality,
    compute_competition_score,
    compute_ranking_score,
    compute_recency_score,
    compute_salary_percentile,
)

_WEIGHTS = RankingWeights(
    resume_match=0.40,
    hiring_probability=0.20,
    salary=0.15,
    recency=0.10,
    company_quality=0.10,
    competition=0.05,
)


def test_weights_sum_to_one() -> None:
    assert _WEIGHTS.total == pytest.approx(1.0)


# --- compute_recency_score -----------------------------------------------------------


def test_recency_score_posted_today_is_max() -> None:
    assert compute_recency_score(days_since_posted=0, recency_decay_days=45) == 100.0


def test_recency_score_at_decay_floor() -> None:
    assert compute_recency_score(days_since_posted=45, recency_decay_days=45) == 10.0


def test_recency_score_beyond_decay_floor_does_not_go_negative() -> None:
    assert compute_recency_score(days_since_posted=900, recency_decay_days=45) == 10.0


def test_recency_score_negative_days_treated_as_zero() -> None:
    assert compute_recency_score(days_since_posted=-3, recency_decay_days=45) == 100.0


# --- compute_salary_percentile -------------------------------------------------------


def test_salary_percentile_neutral_when_salary_missing() -> None:
    assert compute_salary_percentile(None, comparison_salaries=[1.0, 2.0, 3.0]) == 50.0


def test_salary_percentile_neutral_when_too_few_comparisons() -> None:
    assert compute_salary_percentile(100.0, comparison_salaries=[100.0]) == 50.0


def test_salary_percentile_highest_scores_100() -> None:
    result = compute_salary_percentile(300.0, comparison_salaries=[100.0, 200.0, 300.0])
    assert result == 100.0


def test_salary_percentile_lowest_scores_low() -> None:
    result = compute_salary_percentile(100.0, comparison_salaries=[100.0, 200.0, 300.0])
    assert result == pytest.approx(33.33, abs=0.1)


# --- compute_company_quality ----------------------------------------------------------


def test_company_quality_prefers_real_rating() -> None:
    result = compute_company_quality(company_rating=4.5, avg_job_quality_score=10.0)
    assert result == 90.0  # 4.5 / 5.0 * 100


def test_company_quality_falls_back_to_avg_job_quality() -> None:
    result = compute_company_quality(company_rating=None, avg_job_quality_score=72.0)
    assert result == 72.0


def test_company_quality_neutral_when_no_signal() -> None:
    result = compute_company_quality(company_rating=None, avg_job_quality_score=None)
    assert result == 50.0


# --- compute_competition_score --------------------------------------------------------


def test_competition_score_zero_competitors_is_max() -> None:
    assert compute_competition_score(strong_match_count=0) == 100.0


def test_competition_score_saturates_at_zero() -> None:
    assert compute_competition_score(strong_match_count=1000) == 0.0


def test_competition_score_negative_count_treated_as_zero() -> None:
    assert compute_competition_score(strong_match_count=-5) == 100.0


# --- compute_ranking_score -------------------------------------------------------------


def test_ranking_score_all_max_signals_is_100() -> None:
    result = compute_ranking_score(
        resume_match=100.0,
        hiring_probability=100.0,
        salary=100.0,
        recency=100.0,
        company_quality=100.0,
        competition=100.0,
        weights=_WEIGHTS,
    )
    assert result.score == pytest.approx(100.0)


def test_ranking_score_all_zero_signals_is_0() -> None:
    result = compute_ranking_score(
        resume_match=0.0,
        hiring_probability=0.0,
        salary=0.0,
        recency=0.0,
        company_quality=0.0,
        competition=0.0,
        weights=_WEIGHTS,
    )
    assert result.score == pytest.approx(0.0)


def test_ranking_score_missing_resume_match_redistributes_weight() -> None:
    """A user with no resume yet (resume_match=None) shouldn't have every
    job bottom out — the 40% weight redistributes across the other five
    signals rather than being treated as a 0 contribution.
    """
    with_resume = compute_ranking_score(
        resume_match=0.0,  # worst possible match
        hiring_probability=100.0,
        salary=100.0,
        recency=100.0,
        company_quality=100.0,
        competition=100.0,
        weights=_WEIGHTS,
    )
    without_resume = compute_ranking_score(
        resume_match=None,
        hiring_probability=100.0,
        salary=100.0,
        recency=100.0,
        company_quality=100.0,
        competition=100.0,
        weights=_WEIGHTS,
    )
    # Without a resume, the same job (max on every other signal) should
    # rank strictly higher than a job that's actively a bad resume match.
    assert without_resume.score > with_resume.score
    assert without_resume.score == pytest.approx(100.0)


def test_ranking_score_breakdown_preserves_raw_inputs() -> None:
    result = compute_ranking_score(
        resume_match=60.0,
        hiring_probability=70.0,
        salary=80.0,
        recency=90.0,
        company_quality=50.0,
        competition=40.0,
        weights=_WEIGHTS,
    )
    assert result.resume_match == 60.0
    assert result.hiring_probability == 70.0
    assert result.salary == 80.0
    assert result.recency == 90.0
    assert result.company_quality == 50.0
    assert result.competition == 40.0


def test_ranking_score_never_exceeds_bounds() -> None:
    result = compute_ranking_score(
        resume_match=100.0,
        hiring_probability=100.0,
        salary=100.0,
        recency=100.0,
        company_quality=100.0,
        competition=100.0,
        weights=_WEIGHTS,
    )
    assert 0.0 <= result.score <= 100.0
