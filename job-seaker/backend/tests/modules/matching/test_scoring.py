"""Unit tests for pure scoring/boost/narrative logic in `matching.scoring`.

No DB or I/O — these exercise the exact functions the repository/service
delegate scoring decisions to, mirroring `tests/modules/search/test_scoring.py`.
"""

from __future__ import annotations

from app.modules.matching.scoring import (
    build_skill_checklist,
    build_strengths_and_weaknesses,
    compute_match_score,
    country_boost,
    employment_type_boost,
    experience_boost,
    experience_level_from_years,
    salary_boost,
    skill_match,
    work_mode_boost,
)
from app.modules.search.schemas import ExperienceLevel

# --- build_skill_checklist ---------------------------------------------------------


def test_skill_checklist_orders_by_job_skill_list() -> None:
    checklist = build_skill_checklist(
        job_skills=["Python", "FastAPI", "Docker", "AWS", "Kubernetes"],
        matched_skills=["Python", "FastAPI", "Docker"],
        missing_skills=["AWS", "Kubernetes"],
    )
    assert checklist == [
        ("Python", True),
        ("FastAPI", True),
        ("Docker", True),
        ("AWS", False),
        ("Kubernetes", False),
    ]


def test_skill_checklist_case_insensitive_matching() -> None:
    checklist = build_skill_checklist(
        job_skills=["python"], matched_skills=["Python"], missing_skills=[]
    )
    assert checklist == [("python", True)]


def test_skill_checklist_appends_skills_not_on_job_list() -> None:
    # A matched/missing skill absent from the job's own skill list (e.g. a
    # resume-side synonym) still shows up, just appended at the end.
    checklist = build_skill_checklist(
        job_skills=["Python"], matched_skills=["Python", "Django"], missing_skills=[]
    )
    assert checklist == [("Python", True), ("Django", True)]


def test_skill_checklist_no_duplicates() -> None:
    checklist = build_skill_checklist(
        job_skills=["Python", "Python"], matched_skills=["Python"], missing_skills=[]
    )
    assert checklist == [("Python", True)]


def test_skill_checklist_empty_inputs() -> None:
    assert build_skill_checklist(job_skills=[], matched_skills=[], missing_skills=[]) == []

# --- skill_match -----------------------------------------------------------------


def test_skill_match_no_job_skills_scores_full() -> None:
    result = skill_match(["python"], [])
    assert result.score == 100.0
    assert result.matched == []
    assert result.missing == []


def test_skill_match_full_overlap() -> None:
    result = skill_match(["Python", "SQL"], [("Python", "python"), ("SQL", "sql")])
    assert result.score == 100.0
    assert sorted(result.matched) == ["Python", "SQL"]
    assert result.missing == []


def test_skill_match_no_overlap() -> None:
    result = skill_match(["Java"], [("Python", "python"), ("SQL", "sql")])
    assert result.score == 0.0
    assert result.matched == []
    assert sorted(result.missing) == ["Python", "SQL"]


def test_skill_match_partial_overlap() -> None:
    result = skill_match(["python"], [("Python", "python"), ("SQL", "sql")])
    assert result.score == 50.0
    assert result.matched == ["Python"]
    assert result.missing == ["SQL"]


def test_skill_match_is_case_and_whitespace_insensitive() -> None:
    result = skill_match([" Python ", "sql"], [("Python", "python")])
    assert result.score == 100.0


# --- country_boost -----------------------------------------------------------------


def test_country_boost_no_preference_is_neutral() -> None:
    assert country_boost([], "Germany", cap=6.0) is None


def test_country_boost_job_has_no_location() -> None:
    assert country_boost(["Germany"], None, cap=6.0) == 0.0


def test_country_boost_match() -> None:
    assert country_boost(["Germany", "France"], "germany", cap=6.0) == 6.0


def test_country_boost_no_match() -> None:
    assert country_boost(["Germany"], "Spain", cap=6.0) == 0.0


# --- work_mode_boost / employment_type_boost ----------------------------------------


def test_work_mode_boost_no_preference_is_neutral() -> None:
    assert work_mode_boost([], "remote", cap=4.0) is None


def test_work_mode_boost_match() -> None:
    assert work_mode_boost(["remote", "hybrid"], "remote", cap=4.0) == 4.0


def test_work_mode_boost_no_match() -> None:
    assert work_mode_boost(["onsite"], "remote", cap=4.0) == 0.0


def test_employment_type_boost_match() -> None:
    assert employment_type_boost(["full_time"], "full_time", cap=3.0) == 3.0


def test_employment_type_boost_no_match() -> None:
    assert employment_type_boost(["full_time"], "contract", cap=3.0) == 0.0


# --- salary_boost --------------------------------------------------------------------


def test_salary_boost_no_expectation_is_neutral() -> None:
    result = salary_boost(
        expected_min=None,
        expected_max=None,
        expected_currency=None,
        job_min=100_000,
        job_max=120_000,
        job_currency="USD",
        cap=6.0,
    )
    assert result is None


def test_salary_boost_job_has_no_salary() -> None:
    result = salary_boost(
        expected_min=100_000,
        expected_max=150_000,
        expected_currency="USD",
        job_min=None,
        job_max=None,
        job_currency=None,
        cap=6.0,
    )
    assert result == 0.0


def test_salary_boost_currency_mismatch() -> None:
    result = salary_boost(
        expected_min=100_000,
        expected_max=150_000,
        expected_currency="USD",
        job_min=100_000,
        job_max=120_000,
        job_currency="EUR",
        cap=6.0,
    )
    assert result == 0.0


def test_salary_boost_overlapping_ranges() -> None:
    result = salary_boost(
        expected_min=100_000,
        expected_max=150_000,
        expected_currency="USD",
        job_min=140_000,
        job_max=180_000,
        job_currency="USD",
        cap=6.0,
    )
    assert result == 6.0


def test_salary_boost_non_overlapping_ranges() -> None:
    result = salary_boost(
        expected_min=100_000,
        expected_max=120_000,
        expected_currency="USD",
        job_min=200_000,
        job_max=250_000,
        job_currency="USD",
        cap=6.0,
    )
    assert result == 0.0


def test_salary_boost_open_ended_expectation() -> None:
    result = salary_boost(
        expected_min=100_000,
        expected_max=None,
        expected_currency=None,
        job_min=80_000,
        job_max=90_000,
        job_currency=None,
        cap=6.0,
    )
    assert result == 0.0


# --- experience_boost ----------------------------------------------------------------


def test_experience_level_from_years_buckets() -> None:
    assert experience_level_from_years(0) == ExperienceLevel.ENTRY
    assert experience_level_from_years(4) == ExperienceLevel.MID
    assert experience_level_from_years(8) == ExperienceLevel.SENIOR
    assert experience_level_from_years(12) == ExperienceLevel.LEAD
    assert experience_level_from_years(20) == ExperienceLevel.EXECUTIVE


def test_experience_boost_no_years_is_neutral() -> None:
    assert experience_boost(None, "Senior Engineer", cap=6.0) is None


def test_experience_boost_unknown_job_level_is_neutral_zero() -> None:
    assert experience_boost(5, "Software Engineer", cap=6.0) == 0.0


def test_experience_boost_exact_match() -> None:
    # 7 years -> SENIOR bucket; "Senior" title -> SENIOR level. Exact match.
    assert experience_boost(7, "Senior Software Engineer", cap=6.0) == 6.0


def test_experience_boost_adjacent_bucket_half_credit() -> None:
    # 4 years -> MID; "Junior" title -> ENTRY. Adjacent buckets -> half credit.
    assert experience_boost(4, "Junior Software Engineer", cap=6.0) == 3.0


def test_experience_boost_far_bucket_no_credit() -> None:
    # 20 years -> EXECUTIVE; "Junior" title -> ENTRY. Not adjacent -> 0.
    assert experience_boost(20, "Junior Software Engineer", cap=6.0) == 0.0


# --- compute_match_score -------------------------------------------------------------


def test_compute_match_score_weights_and_boosts() -> None:
    score = compute_match_score(
        semantic_score=0.8,
        skill_score=50.0,
        boosts=[6.0, None, 0.0],
        semantic_weight=0.6,
        skill_weight=0.4,
    )
    expected = 0.8 * 100 * 0.6 + 50.0 * 0.4 + 6.0
    assert score == expected


def test_compute_match_score_clamps_to_100() -> None:
    score = compute_match_score(
        semantic_score=1.0,
        skill_score=100.0,
        boosts=[6.0, 6.0, 4.0, 6.0, 3.0],
        semantic_weight=0.6,
        skill_weight=0.4,
    )
    assert score == 100.0


def test_compute_match_score_clamps_to_0() -> None:
    score = compute_match_score(
        semantic_score=0.0,
        skill_score=0.0,
        boosts=[],
        semantic_weight=0.6,
        skill_weight=0.4,
    )
    assert score == 0.0


# --- build_strengths_and_weaknesses ---------------------------------------------------


def test_strengths_and_weaknesses_high_semantic_and_skills() -> None:
    result = build_strengths_and_weaknesses(
        semantic_score=0.9,
        skill_result=skill_match(["python", "sql"], [("Python", "python"), ("SQL", "sql")]),
        boosts={"country": 6.0, "salary": 0.0},
    )
    assert any("strongly align" in s for s in result.strengths)
    assert any("match 2/2" in s for s in result.strengths)
    assert any("preferred countries" in s for s in result.strengths)
    assert any("doesn't fit your expectations" in w for w in result.weaknesses)


def test_strengths_and_weaknesses_low_semantic_and_missing_skills() -> None:
    result = build_strengths_and_weaknesses(
        semantic_score=0.1,
        skill_result=skill_match([], [("Kubernetes", "kubernetes")]),
        boosts={"country": None},
    )
    assert any("doesn't closely match" in w for w in result.weaknesses)
    assert any("Kubernetes" in w for w in result.weaknesses)
    assert result.strengths == []
