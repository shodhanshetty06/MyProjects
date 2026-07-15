"""Base response envelopes shared by every API endpoint in the platform.

Every handler should return one of these (directly or via a subclass) so
API responses have a single, predictable shape for clients to parse.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generic, List, TypeVar

from pydantic import BaseModel, Field

DataT = TypeVar("DataT")


class BaseResponse(BaseModel, Generic[DataT]):
    """Standard success envelope."""

    success: bool = True
    message: str = "OK"
    data: DataT | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorDetail(BaseModel):
    error_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standard error envelope, produced by the global exception handlers."""

    success: bool = False
    error: ErrorDetail
    request_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PaginationMeta(BaseModel):
    page: int
    page_size: int
    total_items: int
    total_pages: int


class PaginatedResponse(BaseModel, Generic[DataT]):
    """Standard envelope for list endpoints."""

    success: bool = True
    message: str = "OK"
    data: List[DataT] = Field(default_factory=list)
    pagination: PaginationMeta
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size
