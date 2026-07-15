"""AI Career Coach module — ORM models.

Two independent concerns:
    CareerInsight        — one generated result for a point-in-time analysis
                            feature (resume review, skill gap, salary
                            prediction, ...). `result` holds the schema
                            defined in `schemas.py` for that `insight_type`
                            as JSONB, so history/detail reads never need to
                            recompute or re-call the model.
    CoachConversation /
    CoachMessage          — the free-form chat assistant's threads.

`user_id` has no foreign key: the `auth` module is still an unimplemented
Phase 0 stub (no `User` table). Ownership is enforced at the service layer
against a trusted `X-User-Id` header — see `app/api/deps.py`.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base, SoftDeleteMixin, TimestampMixin, UUIDPrimaryKeyMixin


class CareerInsightType(str, enum.Enum):
    RESUME_REVIEW = "resume_review"
    CAREER_SUGGESTIONS = "career_suggestions"
    SKILL_GAP_ANALYSIS = "skill_gap_analysis"
    LEARNING_RECOMMENDATIONS = "learning_recommendations"
    SALARY_PREDICTION = "salary_prediction"
    CAREER_ROADMAP = "career_roadmap"
    INTERVIEW_PREPARATION = "interview_preparation"
    JOB_READINESS_SCORE = "job_readiness_score"
    MARKET_TREND_ANALYSIS = "market_trend_analysis"
    COUNTRY_DEMAND = "country_demand"


class CoachMessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"


class CareerInsight(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One generated result for a single-shot analysis feature."""

    __tablename__ = "career_insight"
    __table_args__ = (Index("ix_career_insight_user_type", "user_id", "insight_type"),)

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    insight_type: Mapped[CareerInsightType] = mapped_column(String(40), nullable=False)
    resume_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resume_version.id", ondelete="SET NULL"),
        nullable=True,
    )
    input_params: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    result: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)


class CoachConversation(Base, UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin):
    """A chat thread with the AI career coach assistant."""

    __tablename__ = "coach_conversation"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="New conversation")
    last_message_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    messages: Mapped[list["CoachMessage"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="CoachMessage.created_at",
    )


class CoachMessage(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A single turn (user or assistant) within a `CoachConversation`."""

    __tablename__ = "coach_message"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("coach_conversation.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[CoachMessageRole] = mapped_column(String(10), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    conversation: Mapped["CoachConversation"] = relationship(back_populates="messages")


__all__ = [
    "CareerInsightType",
    "CoachMessageRole",
    "CareerInsight",
    "CoachConversation",
    "CoachMessage",
]
