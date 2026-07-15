"""Search module — application/use-case layer.

Returns fully-built response schemas (not bare ORM objects) directly from
each method, unlike the simpler 1:1 ORM->schema mapping resume_parser/
job_ingestion leave to their routers — here every result also carries a
service-computed `ScoreBreakdown` (keyword/semantic/hybrid) that only this
layer has, so building the response here (rather than splitting ORM object
and score across the router boundary) keeps that fusion in one place.
"""

from __future__ import annotations

import uuid

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.integrations.cache import CacheClient
from app.modules.job_ingestion.models import Job
from app.modules.search.repository import SearchRepository
from app.modules.search.schemas import (
    AutocompleteSuggestion,
    SearchFilters,
    SearchMode,
    SearchResultCompany,
    SearchResultItem,
    SearchResultLocation,
    SearchSortBy,
    SearchSuggestion,
    SkillTag,
    SortDirection,
    TrendingJob,
    TrendingSkill,
)
from app.modules.search.scoring import ScoreBreakdown, classify_experience_level, fuse_scores
from app.schemas.common import PaginationParams
from app.services.ai.service import AIService

logger = get_logger(__name__)


class SearchService:
    """Orchestrates filter/keyword/semantic search, caching, and aggregates."""

    def __init__(
        self,
        repository: SearchRepository,
        ai_service: AIService,
        cache: CacheClient,
    ) -> None:
        self._repository = repository
        self._ai_service = ai_service
        self._cache = cache
        self._settings: Settings = get_settings()

    # --- main search --------------------------------------------------------------

    async def search(
        self, *, filters: SearchFilters, pagination: PaginationParams
    ) -> tuple[list[SearchResultItem], int]:
        cache_key = self._cache.build_key(
            "search:v1:query",
            **filters.model_dump(mode="json"),
            page=pagination.page,
            page_size=pagination.page_size,
        )
        cached = await self._cache.get_json(cache_key)
        if cached is not None:
            items = [SearchResultItem.model_validate(entry) for entry in cached["items"]]
            return items, cached["total"]

        if filters.q is None:
            jobs = await self._repository.list_filtered(
                filters=filters,
                sort_by=filters.sort_by,
                sort_direction=filters.sort_direction,
                offset=pagination.offset,
                limit=pagination.page_size,
            )
            total = await self._repository.count_filtered(filters=filters)
            items = [self._to_result(job, None) for job in jobs]
        else:
            items, total = await self._search_by_query(filters, pagination)

        await self._cache.set_json(
            cache_key,
            {"items": [item.model_dump(mode="json") for item in items], "total": total},
            ttl_seconds=self._settings.SEARCH_CACHE_TTL_SECONDS,
        )
        return items, total

    async def _search_by_query(
        self, filters: SearchFilters, pagination: PaginationParams
    ) -> tuple[list[SearchResultItem], int]:
        # Only called from `search()` when `filters.q` is set (see the
        # `if filters.q is None` branch there); narrowed to `str` here since
        # `filters.q`'s `Optional` type doesn't survive the method boundary.
        assert filters.q is not None
        query = filters.q

        keyword_raw: dict[uuid.UUID, float] = {}
        semantic_raw: dict[uuid.UUID, float] = {}

        if filters.mode in (SearchMode.KEYWORD, SearchMode.HYBRID):
            keyword_raw = dict(
                await self._repository.search_keyword_candidates(
                    filters=filters,
                    query_text=query,
                    limit=self._settings.SEARCH_KEYWORD_CANDIDATE_LIMIT,
                )
            )

        needs_semantic = (
            filters.mode in (SearchMode.SEMANTIC, SearchMode.HYBRID)
            or filters.sort_by == SearchSortBy.MATCH_SCORE
        )
        if needs_semantic:
            embedding = await self._ai_service.generate_embedding(text=query, input_type="query")
            semantic_raw = dict(
                await self._repository.search_semantic_candidates(
                    filters=filters,
                    query_embedding=embedding,
                    limit=self._settings.SEARCH_SEMANTIC_CANDIDATE_LIMIT,
                )
            )

        scores = fuse_scores(
            keyword_raw,
            semantic_raw,
            w_keyword=self._settings.SEARCH_HYBRID_KEYWORD_WEIGHT,
            w_semantic=self._settings.SEARCH_HYBRID_SEMANTIC_WEIGHT,
        )
        if not scores:
            return [], 0

        if filters.sort_by in (SearchSortBy.RELEVANCE, SearchSortBy.MATCH_SCORE):
            return await self._paginate_by_score(scores, filters, pagination)
        return await self._paginate_by_field(scores, filters, pagination)

    async def _paginate_by_score(
        self,
        scores: dict[uuid.UUID, ScoreBreakdown],
        filters: SearchFilters,
        pagination: PaginationParams,
    ) -> tuple[list[SearchResultItem], int]:
        reverse = filters.sort_direction == SortDirection.DESC
        if filters.sort_by == SearchSortBy.MATCH_SCORE:
            key = lambda job_id: scores[job_id].semantic_score or 0.0  # noqa: E731
        else:
            key = lambda job_id: scores[job_id].hybrid_score  # noqa: E731
        ordered_ids = sorted(scores.keys(), key=key, reverse=reverse)
        total = len(ordered_ids)
        page_ids = ordered_ids[pagination.offset : pagination.offset + pagination.page_size]
        jobs_by_id = await self._repository.fetch_jobs_by_ids(page_ids)
        items = [
            self._to_result(jobs_by_id[job_id], scores.get(job_id))
            for job_id in page_ids
            if job_id in jobs_by_id
        ]
        return items, total

    async def _paginate_by_field(
        self,
        scores: dict[uuid.UUID, ScoreBreakdown],
        filters: SearchFilters,
        pagination: PaginationParams,
    ) -> tuple[list[SearchResultItem], int]:
        jobs_by_id = await self._repository.fetch_jobs_by_ids(list(scores.keys()))
        ordered_jobs = _sort_jobs(
            list(jobs_by_id.values()), filters.sort_by, filters.sort_direction
        )
        total = len(ordered_jobs)
        page = ordered_jobs[pagination.offset : pagination.offset + pagination.page_size]
        return [self._to_result(job, scores.get(job.id)) for job in page], total

    def _to_result(self, job: Job, scores: ScoreBreakdown | None) -> SearchResultItem:
        experience_level = None
        if self._settings.SEARCH_EXPERIENCE_LEVEL_HEURISTIC_ENABLED:
            experience_level = classify_experience_level(job.title)
        return SearchResultItem(
            id=job.id,
            title=job.title,
            company=SearchResultCompany.model_validate(job.company),
            location=SearchResultLocation.model_validate(job.location) if job.location else None,
            work_mode=job.work_mode,
            employment_type=job.employment_type,
            salary_min=float(job.salary_min) if job.salary_min is not None else None,
            salary_max=float(job.salary_max) if job.salary_max is not None else None,
            salary_currency=job.salary_currency,
            salary_period=job.salary_period,
            posted_at=job.posted_at,
            skills=[SkillTag.model_validate(skill) for skill in job.skills],
            experience_level=experience_level,
            relevance_score=scores.hybrid_score if scores else None,
            match_score=scores.semantic_score if scores else None,
            keyword_score=scores.keyword_score if scores else None,
        )

    # --- autocomplete / suggestions -------------------------------------------------

    async def autocomplete(self, *, q: str, limit: int) -> list[AutocompleteSuggestion]:
        cache_key = self._cache.build_key("search:v1:autocomplete", q=q, limit=limit)
        cached = await self._cache.get_json(cache_key)
        if cached is not None:
            return [AutocompleteSuggestion.model_validate(entry) for entry in cached]

        if len(q) < self._settings.SEARCH_AUTOCOMPLETE_MIN_CHARS:
            return []

        titles = await self._repository.autocomplete_titles(prefix=q, limit=limit)
        companies = await self._repository.autocomplete_companies(prefix=q, limit=limit)
        skills = await self._repository.autocomplete_skills(prefix=q, limit=limit)
        suggestions = (
            [AutocompleteSuggestion(type="job_title", label=title) for title in titles]
            + [
                AutocompleteSuggestion(type="company", label=name, id=company_id)
                for company_id, name in companies
            ]
            + [
                AutocompleteSuggestion(type="skill", label=name, id=skill_id)
                for skill_id, name in skills
            ]
        )[:limit]

        await self._cache.set_json(
            cache_key,
            [s.model_dump(mode="json") for s in suggestions],
            ttl_seconds=self._settings.SEARCH_AUTOCOMPLETE_CACHE_TTL_SECONDS,
        )
        return suggestions

    async def suggestions(self, *, q: str, limit: int) -> list[SearchSuggestion]:
        """Vocabulary-derived related-term suggestions (skills + job titles).

        Not personalized/click-based — no query-log or view-tracking data
        exists in the schema. Reuses the same trigram infra as autocomplete.
        """
        cache_key = self._cache.build_key("search:v1:suggestions", q=q, limit=limit)
        cached = await self._cache.get_json(cache_key)
        if cached is not None:
            return [SearchSuggestion.model_validate(entry) for entry in cached]

        if len(q) < self._settings.SEARCH_AUTOCOMPLETE_MIN_CHARS:
            return []

        skills = await self._repository.autocomplete_skills(prefix=q, limit=limit)
        titles = await self._repository.autocomplete_titles(prefix=q, limit=limit)
        result = (
            [SearchSuggestion(label=name, type="skill") for _, name in skills]
            + [SearchSuggestion(label=title, type="job_title") for title in titles]
        )[:limit]

        await self._cache.set_json(
            cache_key,
            [s.model_dump(mode="json") for s in result],
            ttl_seconds=self._settings.SEARCH_AUTOCOMPLETE_CACHE_TTL_SECONDS,
        )
        return result

    # --- trending --------------------------------------------------------------------

    async def trending_skills(self, *, window_days: int, limit: int) -> list[TrendingSkill]:
        cache_key = self._cache.build_key(
            "search:v1:trending:skills", window_days=window_days, limit=limit
        )
        cached = await self._cache.get_json(cache_key)
        if cached is not None:
            return [TrendingSkill.model_validate(entry) for entry in cached]

        rows = await self._repository.trending_skills(window_days=window_days, limit=limit)
        result = [
            TrendingSkill(skill_id=skill_id, name=name, job_count=job_count)
            for skill_id, name, job_count in rows
        ]
        await self._cache.set_json(
            cache_key,
            [r.model_dump(mode="json") for r in result],
            ttl_seconds=self._settings.SEARCH_TRENDING_CACHE_TTL_SECONDS,
        )
        return result

    async def trending_jobs(self, *, window_days: int, limit: int) -> list[TrendingJob]:
        cache_key = self._cache.build_key(
            "search:v1:trending:jobs", window_days=window_days, limit=limit
        )
        cached = await self._cache.get_json(cache_key)
        if cached is not None:
            return [TrendingJob.model_validate(entry) for entry in cached]

        jobs = await self._repository.trending_jobs(window_days=window_days, limit=limit)
        result = [
            TrendingJob(
                id=job.id,
                title=job.title,
                company_name=job.company.name,
                posted_at=job.posted_at,
            )
            for job in jobs
        ]
        await self._cache.set_json(
            cache_key,
            [r.model_dump(mode="json") for r in result],
            ttl_seconds=self._settings.SEARCH_TRENDING_CACHE_TTL_SECONDS,
        )
        return result


def _job_sort_value(job: Job, sort_by: SearchSortBy):
    if sort_by == SearchSortBy.SALARY:
        return job.salary_max if job.salary_max is not None else job.salary_min
    if sort_by == SearchSortBy.NEWEST:
        return job.posted_at
    if sort_by == SearchSortBy.COMPANY:
        return job.company.name.lower() if job.company else None
    return None


def _sort_jobs(jobs: list[Job], sort_by: SearchSortBy, direction: SortDirection) -> list[Job]:
    """Sort candidate-pool jobs by a field value, nulls always last —
    matches the `nullslast()` behavior of the pure-filter SQL sort path.
    """
    reverse = direction == SortDirection.DESC
    with_value = [j for j in jobs if _job_sort_value(j, sort_by) is not None]
    without_value = [j for j in jobs if _job_sort_value(j, sort_by) is None]
    with_value.sort(key=lambda j: _job_sort_value(j, sort_by), reverse=reverse)
    return with_value + without_value
