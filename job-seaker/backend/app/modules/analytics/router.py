"""Analytics module — API routes (v1).

Read-only reporting over the caller's own application/resume data (plus
market-wide job-posting aggregates for hiring trends & skill demand). Every
endpoint is scoped to the caller via `CurrentUserIdDep` (trusted
`X-User-Id` header — see `app/api/deps.py`), matching `tracking` and every
other module built against the Phase 0 auth stub.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Response

from app.api.deps import CurrentUserIdDep, DBSessionDep
from app.modules.analytics.repository import AnalyticsRepository
from app.modules.analytics.schemas import AnalyticsDashboardResponse, AnalyticsFilters
from app.modules.analytics.service import AnalyticsService
from app.modules.matching.repository import MatchingRepository
from app.schemas.common import BaseResponse

router = APIRouter(prefix="/analytics", tags=["Analytics"])


def get_analytics_service(session: DBSessionDep) -> AnalyticsService:
    return AnalyticsService(
        repository=AnalyticsRepository(session), matching_repository=MatchingRepository(session)
    )


ServiceDep = Depends(get_analytics_service)
FiltersDep = Annotated[AnalyticsFilters, Depends()]


@router.get("/dashboard", response_model=BaseResponse[AnalyticsDashboardResponse])
async def get_dashboard(
    user_id: CurrentUserIdDep,
    filters: FiltersDep,
    service: AnalyticsService = ServiceDep,
) -> BaseResponse[AnalyticsDashboardResponse]:
    dashboard = await service.get_dashboard(user_id=user_id, filters=filters)
    return BaseResponse(data=dashboard)


@router.get("/export/csv")
async def export_applications_csv(
    user_id: CurrentUserIdDep,
    filters: FiltersDep,
    service: AnalyticsService = ServiceDep,
) -> Response:
    csv_content = await service.export_applications_csv(user_id=user_id, filters=filters)
    filename = f"applications-{datetime.now(UTC).strftime('%Y%m%d')}.csv"
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/export/pdf")
async def export_dashboard_pdf(
    user_id: CurrentUserIdDep,
    filters: FiltersDep,
    service: AnalyticsService = ServiceDep,
) -> Response:
    pdf_bytes = await service.export_dashboard_pdf(user_id=user_id, filters=filters)
    filename = f"analytics-report-{datetime.now(UTC).strftime('%Y%m%d')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
