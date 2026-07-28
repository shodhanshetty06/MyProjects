"""Search module — API routes (v1).

Public, read-only surface (no auth) over ingested `Job` postings: hybrid
keyword/semantic search with filters, autocomplete, related-term
suggestions, and trending aggregates. All GET + query-param bound so
results stay cacheable/bookmarkable.
"""

from __future__ import annotations

import math
import uuid

from fastapi import APIRouter, Depends, Query

from app.api.deps import (
    AIServiceDep,
    CacheDep,
    CurrentUserIdDep,
    DBSessionDep,
    EmbeddingServiceDep,
    OptionalCurrentUserIdDep,
)
from app.modules.matching.repository import MatchingRepository
from app.modules.search.job_feed_service import JobFeedService, SavedJobService
from app.modules.search.nl_query import ParsedSearchQuery, parse_natural_language_query
from app.modules.search.repository import SavedJobRepository, SearchRepository
from app.modules.search.schemas import (
    AutocompleteSuggestion,
    JobFeedFilters,
    JobFeedItem,
    JobSearchHomeSummary,
    SavedJobRead,
    SearchFilters,
    SearchResultItem,
    SearchSuggestion,
    TrendingJob,
    TrendingSkill,
    WhyThisScorePanel,
)
from app.modules.search.service import SearchService
from app.modules.tracking.repository import ApplicationRepository
from app.schemas.common import BaseResponse, PaginatedResponse, PaginationMeta, PaginationParams

router = APIRouter(prefix="/search", tags=["Search"])


def get_search_service(
    session: DBSessionDep, cache: CacheDep, embedding_service: EmbeddingServiceDep
) -> SearchService:
    return SearchService(
        repository=SearchRepository(session),
        embedding_service=embedding_service,
        cache=cache,
    )


ServiceDep = Depends(get_search_service)


def get_saved_job_service(session: DBSessionDep) -> SavedJobService:
    return SavedJobService(
        repository=SavedJobRepository(session), search_repository=SearchRepository(session)
    )


SavedJobServiceDep = Depends(get_saved_job_service)


def get_job_feed_service(
    session: DBSessionDep, cache: CacheDep, embedding_service: EmbeddingServiceDep
) -> JobFeedService:
    return JobFeedService(
        search_service=get_search_service(session, cache, embedding_service),
        search_repository=SearchRepository(session),
        saved_job_repository=SavedJobRepository(session),
        matching_repository=MatchingRepository(session),
        application_repository=ApplicationRepository(session),
        session=session,
    )


JobFeedServiceDep = Depends(get_job_feed_service)


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


@router.get("/parse-query", response_model=BaseResponse[ParsedSearchQuery])
async def parse_query(
    ai_service: AIServiceDep,
    q: str = Query(..., min_length=1, max_length=512),
) -> BaseResponse[ParsedSearchQuery]:
    """"Better Search" — parses one free-text query (e.g. "Find remote
    Python jobs paying above ₹20 LPA") into structured filters + a cleaned
    semantic query. Returns the parse only; the frontend shows it as
    removable filter chips and issues the actual `GET /search` call itself
    (see `nl_query.py` module docstring for why this stays two round trips
    rather than one implicit rewrite).
    """
    parsed = await parse_natural_language_query(q, ai_service=ai_service)
    return BaseResponse(data=parsed)


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


# --- Job Feed (authenticated) --------------------------------------------------------
#
# Everything below composes across `matching`/`tracking`/`workspace` (see
# `job_feed_service.JobFeedService`) rather than duplicating their logic —
# the 5 endpoints above stay the platform's public, unauthenticated search
# surface, unchanged.


@router.get("/feed", response_model=PaginatedResponse[JobFeedItem])
async def get_job_feed(
    user_id: CurrentUserIdDep,
    filters: JobFeedFilters = Depends(),
    pagination: PaginationParams = Depends(),
    service: JobFeedService = JobFeedServiceDep,
) -> PaginatedResponse[JobFeedItem]:
    items, total = await service.get_feed(user_id=user_id, filters=filters, pagination=pagination)
    return PaginatedResponse(
        data=items,
        pagination=PaginationMeta(
            page=pagination.page,
            page_size=pagination.page_size,
            total_items=total,
            total_pages=math.ceil(total / pagination.page_size) if total else 0,
        ),
    )


@router.get("/home-summary", response_model=BaseResponse[JobSearchHomeSummary])
async def get_home_summary(
    user_id: CurrentUserIdDep, service: JobFeedService = JobFeedServiceDep
) -> BaseResponse[JobSearchHomeSummary]:
    summary = await service.get_home_summary(user_id=user_id)
    return BaseResponse(data=summary)


@router.get("/saved-jobs", response_model=PaginatedResponse[JobFeedItem])
async def list_saved_jobs(
    user_id: CurrentUserIdDep,
    pagination: PaginationParams = Depends(),
    service: JobFeedService = JobFeedServiceDep,
) -> PaginatedResponse[JobFeedItem]:
    filters = JobFeedFilters(feed_mode="saved")
    items, total = await service.get_feed(user_id=user_id, filters=filters, pagination=pagination)
    return PaginatedResponse(
        data=items,
        pagination=PaginationMeta(
            page=pagination.page,
            page_size=pagination.page_size,
            total_items=total,
            total_pages=math.ceil(total / pagination.page_size) if total else 0,
        ),
    )


@router.post("/saved-jobs/{job_id}", response_model=BaseResponse[SavedJobRead], status_code=201)
async def save_job(
    job_id: uuid.UUID, user_id: CurrentUserIdDep, service: SavedJobService = SavedJobServiceDep
) -> BaseResponse[SavedJobRead]:
    saved = await service.save(user_id=user_id, job_id=job_id)
    return BaseResponse(message="Job saved.", data=saved)


@router.delete("/saved-jobs/{job_id}", response_model=BaseResponse[None])
async def unsave_job(
    job_id: uuid.UUID, user_id: CurrentUserIdDep, service: SavedJobService = SavedJobServiceDep
) -> BaseResponse[None]:
    await service.unsave(user_id=user_id, job_id=job_id)
    return BaseResponse(message="Job unsaved.")


@router.get("/jobs/{job_id}/why-score", response_model=BaseResponse[WhyThisScorePanel])
async def get_why_score(
    job_id: uuid.UUID, user_id: CurrentUserIdDep, service: JobFeedService = JobFeedServiceDep
) -> BaseResponse[WhyThisScorePanel]:
    panel = await service.get_why_score(user_id=user_id, job_id=job_id)
    return BaseResponse(data=panel)


@router.get("/jobs/{job_id}", response_model=BaseResponse[JobFeedItem])
async def get_job_detail(
    job_id: uuid.UUID,
    user_id: OptionalCurrentUserIdDep,
    service: JobFeedService = JobFeedServiceDep,
) -> BaseResponse[JobFeedItem]:
    item = await service.get_job(user_id=user_id, job_id=job_id)
    return BaseResponse(data=item)
