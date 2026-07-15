"""Tests for `SearchRepository._apply_filters` query-building.

Inspects the compiled SQL of the `Select` object rather than executing
against a live database — these assert *shape* (which clauses/joins get
added for which filter combinations), not query results.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from app.modules.job_ingestion.models import Job, WorkMode
from app.modules.search.repository import SearchRepository
from app.modules.search.schemas import DatePosted, ExperienceLevel, SearchFilters

repository = SearchRepository(session=None)  # type: ignore[arg-type]


def _compiled(stmt) -> str:
    # Compile against the Postgres dialect explicitly — the generic/default
    # compiler renders `.ilike()` as `lower(x) LIKE lower(...)` instead of
    # the native `ILIKE` operator these assertions check for.
    return str(stmt.compile(dialect=postgresql.dialect()))


def test_base_filters_always_exclude_inactive_and_deleted() -> None:
    stmt = repository._apply_filters(select(Job), SearchFilters())
    sql = _compiled(stmt)
    assert "job.status =" in sql
    assert "job.deleted_at IS NULL" in sql


def test_multi_value_work_mode_uses_in_clause() -> None:
    filters = SearchFilters(work_mode=[WorkMode.REMOTE, WorkMode.HYBRID])
    stmt = repository._apply_filters(select(Job), filters)
    sql = _compiled(stmt)
    assert "job.work_mode IN" in sql


def test_skills_or_semantics_produce_single_exists() -> None:
    filters = SearchFilters(skills=["python", "django"], skills_match_all=False)
    stmt = repository._apply_filters(select(Job), filters)
    sql = _compiled(stmt)
    assert sql.count("EXISTS") == 1


def test_skills_and_semantics_produce_one_exists_per_skill() -> None:
    filters = SearchFilters(skills=["python", "django"], skills_match_all=True)
    stmt = repository._apply_filters(select(Job), filters)
    sql = _compiled(stmt)
    assert sql.count("EXISTS") == 2


def test_salary_range_excludes_undisclosed_salary() -> None:
    filters = SearchFilters(salary_min=50000, salary_max=120000)
    stmt = repository._apply_filters(select(Job), filters)
    sql = _compiled(stmt)
    assert "job.salary_min IS NOT NULL" in sql
    assert "job.salary_max IS NOT NULL" in sql
    assert "job.salary_max >=" in sql
    assert "job.salary_min <=" in sql


def test_salary_min_only_does_not_constrain_max() -> None:
    filters = SearchFilters(salary_min=50000)
    stmt = repository._apply_filters(select(Job), filters)
    sql = _compiled(stmt)
    assert "job.salary_max >=" in sql
    assert "job.salary_min <=" not in sql


def test_experience_level_mid_uses_negation_of_all_specific_patterns() -> None:
    filters = SearchFilters(experience_level=[ExperienceLevel.MID])
    stmt = repository._apply_filters(select(Job), filters)
    sql = _compiled(stmt)
    assert "NOT" in sql
    assert "ILIKE" in sql


def test_experience_level_senior_uses_ilike_patterns() -> None:
    filters = SearchFilters(experience_level=[ExperienceLevel.SENIOR])
    stmt = repository._apply_filters(select(Job), filters)
    sql = _compiled(stmt)
    assert "job.title ILIKE" in sql


def test_date_posted_last_7d_filters_posted_at() -> None:
    filters = SearchFilters(date_posted=DatePosted.LAST_7D)
    stmt = repository._apply_filters(select(Job), filters)
    sql = _compiled(stmt)
    assert "job.posted_at >=" in sql


def test_date_posted_any_does_not_filter_posted_at() -> None:
    filters = SearchFilters(date_posted=DatePosted.ANY)
    stmt = repository._apply_filters(select(Job), filters)
    sql = _compiled(stmt)
    assert "job.posted_at >=" not in sql


def test_location_filters_join_location_table() -> None:
    filters = SearchFilters(city=["Austin"])
    stmt = repository._apply_filters(select(Job), filters)
    sql = _compiled(stmt)
    assert "JOIN location" in sql
    assert "location.city IN" in sql


def test_company_name_filter_joins_company() -> None:
    filters = SearchFilters(company_name="Acme")
    stmt = repository._apply_filters(select(Job), filters)
    sql = _compiled(stmt)
    assert "JOIN company" in sql
    assert "company.name ILIKE" in sql
