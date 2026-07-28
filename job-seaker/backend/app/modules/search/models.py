"""Search module — ORM models.

Search itself owns no `Job`/`Company`/`Location`/`Skill`/`JobEmbedding`
tables — it queries `job_ingestion`'s directly (see `repository.py`). This
module's *own* first table is `SavedJob`: a user's bookmark of a job,
persisted server-side so saves sync across devices (previously 100%
client-only Zustand state — see `frontend/src/stores/pinsStore.ts`).

`user_id` has no foreign key, mirroring `matching.MatchPreference` and
`tracking.Application` — the `auth`/`user_profile` modules are still
unimplemented Phase 0 stubs, so ownership is enforced at the service layer
against the authenticated caller's id (see `app/api/deps.py`).
"""

from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base_class import Base, TimestampMixin, UUIDPrimaryKeyMixin


class SavedJob(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One user's bookmark of one `job_ingestion.Job`.

    Unique per `(user_id, job_id)` — saving an already-saved job is a no-op
    (idempotent get-or-create at the service layer), not a duplicate row.
    """

    __tablename__ = "saved_job"
    __table_args__ = (UniqueConstraint("user_id", "job_id", name="uq_saved_job_user_job"),)

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("job.id", ondelete="CASCADE"), nullable=False, index=True
    )


__all__ = ["SavedJob"]
