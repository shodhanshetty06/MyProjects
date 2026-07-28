"""add performance indexes

Composite/covering indexes for query patterns that are structurally
correct today but only visible at scale (the dev DB has tens of rows per
table, so the planner picks seq scans regardless — these are added based
on the WHERE/ORDER BY shape of the actual repository queries, not on
`EXPLAIN` output from the tiny dev dataset):

  - `resume_job_match` is read by `MatchingRepository.list_matches_for_resume`
    (WHERE resume_version_id = ... ORDER BY match_score DESC) and
    `list_matches_for_user` (WHERE user_id = ... AND status = 'computed'
    ORDER BY match_score DESC) — both filter-then-sort on a column with no
    existing index, so postgres must sort every matching row on every call.
  - `notification` is read by `NotificationRepository.list_filtered` /
    `count_filtered` (WHERE user_id = ... [AND read_at IS NULL]) — only a
    single-column index on `user_id` exists today, so the unread-only path
    (the common case: notification bell / workspace summary) still has to
    revisit every one of a user's notifications instead of only the unread
    ones.
  - `tracked_email` is read by `TrackedEmailRepository.list_filtered`
    (joins to `gmail_account`, ORDER BY received_at DESC) — a composite
    index lets postgres satisfy the per-mailbox listing without a separate
    sort step as a mailbox accumulates history.
  - `job` is listed unfiltered (no company_id/status given) by
    `JobRepository.list_all` (GET /job-ingestion/jobs) ordered by
    created_at DESC — `job` is the table expected to grow the largest in
    this platform (continuous ingestion across many companies), and today
    it has no index at all on `created_at`.

Revision ID: 0015_performance_indexes
Revises: 0014_vector_dimensions_384
Create Date: 2026-07-24
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0015_performance_indexes"
down_revision: Union[str, None] = "0014_vector_dimensions_384"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_resume_job_match_version_score",
        "resume_job_match",
        ["resume_version_id", "match_score"],
    )
    op.create_index(
        "ix_resume_job_match_user_status_score",
        "resume_job_match",
        ["user_id", "status", "match_score"],
    )
    op.create_index(
        "ix_notification_user_read_at",
        "notification",
        ["user_id", "read_at"],
    )
    op.create_index(
        "ix_tracked_email_account_received_at",
        "tracked_email",
        ["gmail_account_id", "received_at"],
    )
    op.create_index("ix_job_created_at", "job", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_job_created_at", table_name="job")
    op.drop_index("ix_tracked_email_account_received_at", table_name="tracked_email")
    op.drop_index("ix_notification_user_read_at", table_name="notification")
    op.drop_index("ix_resume_job_match_user_status_score", table_name="resume_job_match")
    op.drop_index("ix_resume_job_match_version_score", table_name="resume_job_match")
