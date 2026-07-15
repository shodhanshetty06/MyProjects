"""Search module — API routes (v1).

Public, read-only surface (no auth) over ingested `Job` postings: hybrid
keyword/semantic search with filters, autocomplete, related-term
suggestions, and trending aggregates. All GET + query-param bound so
results stay cacheable/bookmarkable.
"""

from __future__ import annotations

import math

from fastapi import APIRouter, Depends, Query

from app.api.deps import CacheDep, DBSessionDep
from app.modules.search.repository import SearchRepository
from app.modules.search.schemas import (
    AutocompleteSuggestion,
    SearchFilters,
    SearchResultItem,
    SearchSuggestion,
    TrendingJob,
    TrendingSkill,
)
from app.modules.search.service import SearchService
from app.schemas.common import BaseResponse, PaginatedResponse, PaginationMeta, PaginationParams
from app.services.ai.service import AIService

router = APIRouter(prefix="/search", tags=["Search"])


def get_search_service(session: DBSessionDep, cache: CacheDep) -> SearchService:
    return SearchService(
        repository=SearchRepository(session),
        ai_service=AIService(),
        cache=cache,
    )


ServiceDep = Depends(get_search_service)


@router.get("", response_model=PaginatedResponse[SearchResultItem])
async def search_jobs(
    filters: SearchFilters = Depends(),
    pagination: PaginationParams = Depends(),
    service: SearchService = ServiceDep,
) -> PaginatedResponse[SearchResultItem]:
    items, total = await service.search(filters=filters, pagination=pagination)
    return PaginatedResponse(
        data=items,
        pagination=PaginationMeta(
            page=pagination.page,
            page_size=pagination.page_size,
            total_items=total,
            total_pages=math.ceil(total / pagination.page_size) if total else 0,
        ),
    )


@router.get("/autocomplete", response_model=BaseResponse[list[AutocompleteSuggestion]])
async def autocomplete(
    q: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(default=10, ge=1, le=20),
    service: SearchService = ServiceDep,
) -> BaseResponse[list[AutocompleteSuggestion]]:
    suggestions = await service.autocomplete(q=q, limit=limit)
    return BaseResponse(data=suggestions)


@router.get("/suggestions", response_model=BaseResponse[list[SearchSuggestion]])
async def suggestions(
    q: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(default=8, ge=1, le=20),
    service: SearchService = ServiceDep,
) -> BaseResponse[list[SearchSuggestion]]:
    result = await service.suggestions(q=q, limit=limit)
    return BaseResponse(data=result)


@router.get("/trending/skills", response_model=BaseResponse[list[TrendingSkill]])
async def trending_skills(
    window_days: int = Query(default=30, ge=1, le=90),
    limit: int = Query(default=20, ge=1, le=50),
    service: SearchService = ServiceDep,
) -> BaseResponse[list[TrendingSkill]]:
    result = await service.trending_skills(window_days=window_days, limit=limit)
    return BaseResponse(data=result)


@router.get("/trending/jobs", response_model=BaseResponse[list[TrendingJob]])
async def trending_jobs(
    window_days: int = Query(default=30, ge=1, le=90),
    limit: int = Query(default=20, ge=1, le=50),
    service: SearchService = ServiceDep,
) -> BaseResponse[list[TrendingJob]]:
    result = await service.trending_jobs(window_days=window_days, limit=limit)
    return BaseResponse(data=result)
