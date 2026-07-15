"""Unit tests for pure ranking/fusion/heuristic logic in `search.scoring`.

No DB or I/O — these exercise the exact functions the repository/service
delegate ranking decisions to.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from app.modules.search.schemas import DatePosted, ExperienceLevel
from app.modules.search.scoring import (
    classify_experience_level,
    date_posted_cutoff,
    fuse_scores,
    matches_experience_level,
    normalize_keyword_scores,
)

JOB_A = uuid.uuid4()
JOB_B = uuid.uuid4()
JOB_C = uuid.uuid4()


def test_normalize_keyword_scores_empty() -> None:
    assert normalize_keyword_scores({}) == {}


def test_normalize_keyword_scores_single_value() -> None:
    result = normalize_keyword_scores({JOB_A: 0.42})
    assert result == {JOB_A: 1.0}


def test_normalize_keyword_scores_all_equal() -> None:
    result = normalize_keyword_scores({JOB_A: 0.5, JOB_B: 0.5})
    assert result == {JOB_A: 1.0, JOB_B: 1.0}


def test_normalize_keyword_scores_spread() -> None:
    result = normalize_keyword_scores({JOB_A: 0.0, JOB_B: 0.5, JOB_C: 1.0})
    assert result[JOB_A] == 0.0
    assert result[JOB_B] == 0.5
    assert result[JOB_C] == 1.0


def test_fuse_scores_keyword_only() -> None:
    result = fuse_scores({JOB_A: 1.0}, {}, w_keyword=0.4, w_semantic=0.6)
    breakdown = result[JOB_A]
    assert breakdown.semantic_score is None
    assert breakdown.keyword_score == 1.0
    assert breakdown.hybrid_score == 0.4


def test_fuse_scores_semantic_only() -> None:
    result = fuse_scores({}, {JOB_A: 0.8}, w_keyword=0.4, w_semantic=0.6)
    breakdown = result[JOB_A]
    assert breakdown.keyword_score is None
    assert breakdown.semantic_score == 0.8
    assert breakdown.hybrid_score == 0.6 * 0.8


def test_fuse_scores_overlapping_candidate() -> None:
    result = fuse_scores({JOB_A: 1.0}, {JOB_A: 0.5}, w_keyword=0.4, w_semantic=0.6)
    breakdown = result[JOB_A]
    assert breakdown.keyword_score == 1.0
    assert breakdown.semantic_score == 0.5
    assert breakdown.hybrid_score == 0.4 * 1.0 + 0.6 * 0.5


def test_fuse_scores_weight_sensitivity_changes_ranking() -> None:
    keyword = {JOB_A: 1.0, JOB_B: 0.0}
    semantic = {JOB_A: 0.0, JOB_B: 1.0}

    keyword_favored = fuse_scores(keyword, semantic, w_keyword=0.9, w_semantic=0.1)
    assert keyword_favored[JOB_A].hybrid_score > keyword_favored[JOB_B].hybrid_score

    semantic_favored = fuse_scores(keyword, semantic, w_keyword=0.1, w_semantic=0.9)
    assert semantic_favored[JOB_B].hybrid_score > semantic_favored[JOB_A].hybrid_score


def test_classify_experience_level_senior() -> None:
    assert classify_experience_level("Senior Backend Engineer") == ExperienceLevel.SENIOR


def test_classify_experience_level_entry() -> None:
    assert classify_experience_level("Junior Data Analyst") == ExperienceLevel.ENTRY


def test_classify_experience_level_executive() -> None:
    assert classify_experience_level("VP of Engineering") == ExperienceLevel.EXECUTIVE


def test_classify_experience_level_internship() -> None:
    assert classify_experience_level("Marketing Intern") == ExperienceLevel.INTERNSHIP


def test_classify_experience_level_unknown_returns_none() -> None:
    assert classify_experience_level("Software Engineer") is None


def test_matches_experience_level_mid_is_negation_of_others() -> None:
    assert matches_experience_level("Software Engineer", ExperienceLevel.MID) is True
    assert matches_experience_level("Senior Software Engineer", ExperienceLevel.MID) is False


def test_date_posted_cutoff_any_returns_none() -> None:
    assert date_posted_cutoff(DatePosted.ANY) is None


def test_date_posted_cutoff_last_24h() -> None:
    now = datetime(2026, 7, 11, tzinfo=UTC)
    cutoff = date_posted_cutoff(DatePosted.LAST_24H, now=now)
    assert cutoff == now - timedelta(hours=24)


def test_date_posted_cutoff_last_30d() -> None:
    now = datetime(2026, 7, 11, tzinfo=UTC)
    cutoff = date_posted_cutoff(DatePosted.LAST_30D, now=now)
    assert cutoff == now - timedelta(days=30)
