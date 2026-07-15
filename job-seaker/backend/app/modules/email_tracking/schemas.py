"""Email Tracking module — Pydantic request/response contracts."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, model_validator

from app.modules.email_tracking.models import (
    EmailClassification,
    EmailMatchStatus,
    GmailAccountStatus,
)
from app.modules.tracking.models import ApplicationStatus


class GmailConnectResponse(BaseModel):
    authorization_url: str


class GmailAccountRead(BaseModel):
    id: uuid.UUID
    email_address: str
    status: GmailAccountStatus
    history_id: str | None
    last_synced_at: datetime | None
    last_sync_error: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class TrackedEmailFilters(BaseModel):
    """Bound as a query-param dependency: `filters: TrackedEmailFilters = Depends()`."""

    match_status: EmailMatchStatus | None = None
    application_id: uuid.UUID | None = None


class TrackedEmailRead(BaseModel):
    id: uuid.UUID
    gmail_account_id: uuid.UUID
    application_id: uuid.UUID | None
    sender: str
    subject: str
    snippet: str
    received_at: datetime
    classification: EmailClassification
    confidence: float
    match_status: EmailMatchStatus
    created_at: datetime

    model_config = {"from_attributes": True}


class TrackedEmailDetail(TrackedEmailRead):
    matched_status_history_id: uuid.UUID | None
    raw_metadata: dict[str, Any]


class TrackedEmailResolveRequest(BaseModel):
    """Manual review resolution for a NEEDS_REVIEW/UNMATCHED `TrackedEmail`.

    Either dismiss it (no application/status change) or confirm the match by
    supplying the application to update and the status to set.
    """

    dismiss: bool = False
    application_id: uuid.UUID | None = None
    status: ApplicationStatus | None = None

    @model_validator(mode="after")
    def _validate(self) -> TrackedEmailResolveRequest:
        if self.dismiss:
            if self.application_id is not None or self.status is not None:
                raise ValueError("dismiss cannot be combined with application_id/status")
        elif self.application_id is None or self.status is None:
            raise ValueError("application_id and status are required unless dismiss=true")
        return self


__all__ = [
    "GmailConnectResponse",
    "GmailAccountRead",
    "TrackedEmailFilters",
    "TrackedEmailRead",
    "TrackedEmailDetail",
    "TrackedEmailResolveRequest",
]
