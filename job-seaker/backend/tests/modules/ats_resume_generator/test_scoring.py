"""Unit tests for `app.modules.ats_resume_generator.scoring` — pure functions,
no DB/HTTP/AI I-O required.
"""
from __future__ import annotations

import pytest

from app.modules.ats_resume_generator import scoring


def test_keyword_coverage_matches_case_insensitively() -> None:
    result = scoring.keyword_coverage(
        ["Python", "Kubernetes", "GraphQL"],
        "Experienced backend engineer skilled in python and kubernetes clusters.",
    )
    assert result.matched == ["Python", "Kubernetes"]
    assert result.missing == ["GraphQL"]
    assert round(result.score, 2) == round(2 / 3 * 100, 2)


def test_keyword_coverage_empty_keywords_scores_full() -> None:
    result = scoring.keyword_coverage([], "anything")
    assert result.matched == []
    assert result.missing == []
    assert result.score == 100.0


def test_keyword_coverage_deduplicates_keywords() -> None:
    result = scoring.keyword_coverage(["Python", "python", "Python"], "I write python daily.")
    assert result.matched == ["Python"]
    assert result.score == 100.0


def test_build_resume_search_text_flattens_all_sections() -> None:
    text = scoring.build_resume_search_text(
        summary="Backend engineer.",
        skills=["Python", "SQL"],
        projects=[
            {
                "name": "Order Service",
                "description": "Built a payments pipeline.",
                "technologies": ["Kafka"],
                "highlights": ["Cut latency by 30%"],
            }
        ],
        experience=[
            {"title": "Engineer", "description": "Led backend team.", "highlights": ["Shipped API"]}
        ],
        education=[{"degree": "BSc", "field_of_study": "Computer Science"}],
        certifications=[{"name": "AWS Certified"}],
    )
    for expected in (
        "Backend engineer.",
        "Python",
        "Order Service",
        "Kafka",
        "Cut latency by 30%",
        "Led backend team.",
        "Shipped API",
        "Computer Science",
        "AWS Certified",
    ):
        assert expected in text


def test_structure_score_rewards_present_sections_and_quantified_bullets() -> None:
    result = scoring.structure_score(
        has_summary=True,
        has_skills=True,
        has_experience=True,
        has_education=True,
        has_projects=True,
        bullet_highlights=["Cut latency by 30%", "Led a team of 5", "Improved reliability"],
    )
    assert result.checks["has_summary"] is True
    assert result.score == pytest.approx(75.0 + (2 / 3) * 25.0)
    assert result.score <= 100.0


def test_structure_score_zero_sections_and_no_bullets() -> None:
    result = scoring.structure_score(
        has_summary=False,
        has_skills=False,
        has_experience=False,
        has_education=False,
        has_projects=False,
        bullet_highlights=[],
    )
    assert result.score == 0.0
    assert result.checks["quantified_bullet_ratio"] == 0.0


def test_compute_ats_score_weights_and_clamps() -> None:
    score = scoring.compute_ats_score(
        keyword_score=80.0, structure_score_value=60.0, keyword_weight=0.6, structure_weight=0.4
    )
    assert score == 80.0 * 0.6 + 60.0 * 0.4

    clamped_high = scoring.compute_ats_score(
        keyword_score=100.0, structure_score_value=100.0, keyword_weight=0.8, structure_weight=0.8
    )
    assert clamped_high == 100.0

    clamped_low = scoring.compute_ats_score(
        keyword_score=0.0, structure_score_value=0.0, keyword_weight=0.6, structure_weight=0.4
    )
    assert clamped_low == 0.0
