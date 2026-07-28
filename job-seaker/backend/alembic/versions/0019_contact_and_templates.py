"""Resume contact header + ATS resume template/skill-grouping columns

Adds:
  - resume_parsed_data.contact (JSONB) — extracted header/contact block
    (full_name, phone, email, linkedin_url, github_url, portfolio_url,
    location).
  - ats_resume_version.contact (JSONB) — verbatim copy of the above, never
    sent to or rewritten by Claude.
  - ats_resume_version.template_id (String) — selected visual template for
    DOCX/PDF rendering.
  - ats_resume_version.grouped_skills (JSONB) — AI-organized skill
    categories, a strict repartition of optimized_skills.
  - ats_resume_version.validation_warnings (JSONB) — non-blocking issues
    from the pre-export validator.

Revision ID: 0019_contact_and_templates
Revises: 0018_narrative_events_agent
Create Date: 2026-07-27
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0019_contact_and_templates"
down_revision: Union[str, None] = "0018_narrative_events_agent"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "resume_parsed_data",
        sa.Column("contact", postgresql.JSONB(), nullable=False, server_default="{}"),
    )

    op.add_column(
        "ats_resume_version",
        sa.Column("contact", postgresql.JSONB(), nullable=False, server_default="{}"),
    )
    op.add_column(
        "ats_resume_version",
        sa.Column(
            "template_id", sa.String(length=50), nullable=False, server_default="modern_ats"
        ),
    )
    op.add_column(
        "ats_resume_version",
        sa.Column("grouped_skills", postgresql.JSONB(), nullable=False, server_default="[]"),
    )
    op.add_column(
        "ats_resume_version",
        sa.Column("validation_warnings", postgresql.JSONB(), nullable=False, server_default="[]"),
    )


def downgrade() -> None:
    op.drop_column("ats_resume_version", "validation_warnings")
    op.drop_column("ats_resume_version", "grouped_skills")
    op.drop_column("ats_resume_version", "template_id")
    op.drop_column("ats_resume_version", "contact")
    op.drop_column("resume_parsed_data", "contact")
