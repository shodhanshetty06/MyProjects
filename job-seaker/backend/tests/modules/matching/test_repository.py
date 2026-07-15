"""Tests for `MatchingRepository`'s query-building helpers.

Inspects compiled SQL of `Select` objects rather than executing against a
live database — same no-DB style as
`tests/modules/search/test_repository_filters.py`
(`repository = SearchRepository(session=None)`), since a real Postgres +
pgvector instance isn't available in this sandbox.
"""

from __future__ import annotations

import uuid

from sqlalchemy.dialects import postgresql

from app.modules.matching.models import RecommendationType
from app.modules.matching.repository import _RECOMMENDATION_TARGET_FIELD, MatchingRepository

USER_ID = uuid.uuid4()
RESUME_VERSION_ID = uuid.uuid4()


def _compiled(stmt) -> str:
    # Compile against the Postgres dialect explicitly, consistent with the
    # search module's repository tests (renders `<=`, `ILIKE`, etc. natively).
    return str(stmt.compile(dialect=postgresql.dialect()))


def test_semantic_candidate_stmt_filters_active_and_orders_by_distance() -> None:
    stmt = MatchingRepository._semantic_candidate_stmt(embedding=[0.1, 0.2, 0.3], limit=50)
    sql = _compiled(stmt)
    assert "job.status =" in sql
    assert "job.deleted_at IS NULL" in sql
    assert "job_embedding" in sql
    assert "ORDER BY" in sql
    assert "LIMIT" in sql


def test_semantic_candidate_stmt_respects_limit() -> None:
    # A real 1024-dim vector is required here (unlike the other assertions,
    # which don't render literal binds) — pgvector's literal compiler
    # validates the embedding's dimensionality against `VECTOR_DIMENSIONS`.
    embedding = [0.1] * 1024
    stmt = MatchingRepository._semantic_candidate_stmt(embedding=embedding, limit=7)
    compiled = stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    assert "LIMIT 7" in str(compiled)


def test_matches_for_resume_base_stmt_scopes_to_resume_and_user() -> None:
    stmt = MatchingRepository._matches_for_resume_base_stmt(
        resume_version_id=RESUME_VERSION_ID, user_id=USER_ID
    )
    sql = _compiled(stmt)
    assert "resume_job_match.resume_version_id" in sql
    assert "resume_job_match.user_id" in sql


def test_recommendation_target_field_mapping_is_exhaustive() -> None:
    assert _RECOMMENDATION_TARGET_FIELD[RecommendationType.JOB] == "job_id"
    assert _RECOMMENDATION_TARGET_FIELD[RecommendationType.COMPANY] == "company_id"
    assert _RECOMMENDATION_TARGET_FIELD[RecommendationType.SKILL] == "skill_id"
    assert set(_RECOMMENDATION_TARGET_FIELD) == set(RecommendationType)
