"""Analytics module — ORM models.

This module owns no tables of its own. It is a read-only reporting layer
that aggregates data already owned by `tracking`, `resume_parser`,
`ats_resume_generator`, and `job_ingestion` — see
`app.modules.analytics.repository` for the cross-module queries.
"""
from app.db.base_class import Base  # noqa: F401
