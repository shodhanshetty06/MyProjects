"""Analytics module — application/use-case layer.

Assembles the dashboard payload from `AnalyticsRepository`'s aggregation
queries, fills trend series in with zero-count buckets so charts render
continuous axes, and builds the CSV/PDF export artifacts.
"""
from __future__ import annotations

import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta

from app.core.config import get_settings
from app.core.logging import get_logger
from app.modules.analytics.reports import (
    AnalyticsReportContent,
    generate_analytics_pdf,
    generate_applications_csv,
)
from app.modules.analytics.repository import AnalyticsRepository
from app.modules.analytics.schemas import (
    AcceptanceRatePoint,
    AcceptanceRateSeries,
    AnalyticsDashboardResponse,
    AnalyticsFilters,
    AnalyticsSummary,
    ApplicationHeatmap,
    ATSResumePerformance,
    ATSScoreTrendPoint,
    CompanyOutcome,
    CountryCount,
    ExportApplicationRow,
    HeatmapCell,
    HiringCompanyCount,
    HiringTrendPoint,
    HiringTrends,
    MatchedCompanyCount,
    ResponseTimePoint,
    ResponseTimeSeries,
    ResumePerformance,
    ScoreDistributionBucket,
    SkillCount,
    SkillDemandItem,
    TrendInterval,
    TrendPoint,
    TrendSeries,
)
from app.modules.matching.repository import MatchingRepository
from app.modules.tracking.models import ApplicationStatus

logger = get_logger(__name__)

_DEFAULT_WINDOW_DAYS = 90
_HEATMAP_WINDOW_DAYS = 365
_INTERVIEW_STATUSES = (ApplicationStatus.INTERVIEW, ApplicationStatus.ASSESSMENT)
_OFFER_STATUSES = (ApplicationStatus.OFFER, ApplicationStatus.ACCEPTED)
_SCORE_BANDS = [
    ("0-59", 0, 59),
    ("60-69", 60, 69),
    ("70-79", 70, 79),
    ("80-89", 80, 89),
    ("90-100", 90, 100),
]


def _truncate(dt: datetime, interval: TrendInterval) -> datetime:
    dt = dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
    if interval == TrendInterval.DAY:
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if interval == TrendInterval.MONTH:
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    day = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return day - timedelta(days=day.weekday())  # Monday, matching Postgres date_trunc('week', ...)


def _next_bucket(current: datetime, interval: TrendInterval) -> datetime:
    if interval == TrendInterval.DAY:
        return current + timedelta(days=1)
    if interval == TrendInterval.WEEK:
        return current + timedelta(weeks=1)
    if current.month == 12:
        return current.replace(year=current.year + 1, month=1)
    return current.replace(month=current.month + 1)


def _bucket_starts(date_from: datetime, date_to: datetime, interval: TrendInterval) -> list[datetime]:
    start = _truncate(date_from, interval)
    end = _truncate(date_to, interval)
    buckets = [start]
    current = start
    while current < end:
        current = _next_bucket(current, interval)
        buckets.append(current)
    return buckets


def _fill_trend(
    rows: list[tuple[datetime, int]], *, date_from: datetime, date_to: datetime, interval: TrendInterval
) -> list[TrendPoint]:
    counts = {_truncate(bucket, interval): count for bucket, count in rows}
    return [
        TrendPoint(bucket=bucket, count=counts.get(bucket, 0))
        for bucket in _bucket_starts(date_from, date_to, interval)
    ]


def _guess_country(location: str | None) -> str:
    """Best-effort country extraction for manually-tracked applications.

    Applications linked to an ingested `Job` carry a normalized
    `Location.country`; everything else only has the free-text `location`
    the user typed, so we fall back to its last comma-separated segment
    (e.g. "San Francisco, CA, USA" -> "USA").
    """
    if not location or not location.strip():
        return "Unknown"
    parts = [p.strip() for p in location.split(",") if p.strip()]
    return parts[-1] if parts else "Unknown"


def _count_skills(skill_lists: list[list[str]]) -> tuple[Counter, dict[str, str]]:
    """Count how many (current) resumes contain each skill.

    Dedupes within a single resume so a repeated skill on one resume only
    counts once, then counts across resumes — meaningful once a user has
    more than one tailored resume.
    """
    counter: Counter[str] = Counter()
    display_names: dict[str, str] = {}
    for skills in skill_lists:
        seen: set[str] = set()
        for skill in skills:
            if not skill or not skill.strip():
                continue
            key = skill.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            display_names.setdefault(key, skill.strip())
            counter[key] += 1
    return counter, display_names


class AnalyticsService:
    """Builds the analytics dashboard and its CSV/PDF exports.

    `matching_repository` composes in the Job Search integration fields
    (`average_match_score`, `top_matched_companies`) the same way
    `WorkspaceService` composes across modules — read via
    `MatchingRepository.list_matches_for_user`/`get_jobs_by_ids`, not a new
    duplicate scoring/aggregation query.
    """

    def __init__(
        self, repository: AnalyticsRepository, matching_repository: MatchingRepository
    ) -> None:
        self.repository = repository
        self.matching_repository = matching_repository
        self._settings = get_settings()

    def _resolve_window(self, filters: AnalyticsFilters) -> tuple[datetime, datetime]:
        now = datetime.now(UTC)
        date_to = filters.date_to or now
        date_from = filters.date_from or (date_to - timedelta(days=_DEFAULT_WINDOW_DAYS))
        return date_from, date_to

    async def get_dashboard(
        self, *, user_id: uuid.UUID, filters: AnalyticsFilters
    ) -> AnalyticsDashboardResponse:
        date_from, date_to = self._resolve_window(filters)
        interval = filters.interval
        top_n = filters.top_n

        summary = await self._build_summary(user_id=user_id, date_from=date_from, date_to=date_to)
        applications_trend = await self._applications_trend(user_id, date_from, date_to, interval)
        interviews_trend = await self._status_trend(
            user_id, _INTERVIEW_STATUSES, date_from, date_to, interval
        )
        offers_trend = await self._status_trend(user_id, (ApplicationStatus.OFFER,), date_from, date_to, interval)
        acceptance_rate = await self._acceptance_rate(user_id, date_from, date_to, interval)
        resume_performance = await self._resume_performance(user_id, date_from, date_to, interval)
        ats_resume_performance = await self._ats_resume_performance(user_id, date_from, date_to, interval)
        top_skills = await self._top_skills(user_id, top_n)
        top_companies = await self._top_companies(user_id, date_from, date_to, top_n)
        top_countries = await self._top_countries(user_id, date_from, date_to, top_n)
        hiring_trends = await self._hiring_trends(date_from, date_to, interval, top_n)
        skill_demand = await self._skill_demand(user_id, top_n)
        response_time = await self._response_time(user_id, date_from, date_to, interval)
        heatmap = await self._heatmap(user_id, date_from, date_to)
        saved_jobs_count = await self.repository.count_saved_jobs(user_id=user_id)
        average_match_score, top_matched_companies = await self._match_summary(user_id, top_n)

        return AnalyticsDashboardResponse(
            generated_at=datetime.now(UTC),
            date_from=filters.date_from,
            date_to=filters.date_to,
            interval=interval,
            summary=summary,
            applications_trend=applications_trend,
            interviews_trend=interviews_trend,
            offers_trend=offers_trend,
            acceptance_rate=acceptance_rate,
            resume_performance=resume_performance,
            ats_resume_performance=ats_resume_performance,
            top_skills=top_skills,
            top_companies=top_companies,
            top_countries=top_countries,
            hiring_trends=hiring_trends,
            skill_demand=skill_demand,
            response_time=response_time,
            heatmap=heatmap,
            saved_jobs_count=saved_jobs_count,
            average_match_score=average_match_score,
            top_matched_companies=top_matched_companies,
        )

    # --- section builders --------------------------------------------------------------

    async def _build_summary(
        self, *, user_id: uuid.UUID, date_from: datetime, date_to: datetime
    ) -> AnalyticsSummary:
        status_counts = await self.repository.status_counts_in_range(
            user_id=user_id, date_from=date_from, date_to=date_to
        )
        total = sum(status_counts.values())
        interview_plus = sum(
            status_counts.get(s, 0) for s in (*_INTERVIEW_STATUSES, *_OFFER_STATUSES)
        )
        offer_plus = sum(status_counts.get(s, 0) for s in _OFFER_STATUSES)
        accepted = status_counts.get(ApplicationStatus.ACCEPTED, 0)

        response_rows = await self.repository.response_time_series(
            user_id=user_id, date_from=date_from, date_to=date_to, interval=TrendInterval.MONTH
        )
        weighted_total = sum(avg * n for _, avg, n in response_rows if avg is not None)
        sample_total = sum(n for _, avg, n in response_rows if avg is not None)

        return AnalyticsSummary(
            total_applications=total,
            interview_rate=round(interview_plus / total, 4) if total else 0.0,
            offer_rate=round(offer_plus / total, 4) if total else 0.0,
            acceptance_rate=round(accepted / offer_plus, 4) if offer_plus else 0.0,
            average_response_days=round(weighted_total / sample_total, 2) if sample_total else None,
        )

    async def _applications_trend(
        self, user_id: uuid.UUID, date_from: datetime, date_to: datetime, interval: TrendInterval
    ) -> TrendSeries:
        rows = await self.repository.applications_trend(
            user_id=user_id, date_from=date_from, date_to=date_to, interval=interval
        )
        return TrendSeries(
            interval=interval, points=_fill_trend(rows, date_from=date_from, date_to=date_to, interval=interval)
        )

    async def _status_trend(
        self,
        user_id: uuid.UUID,
        to_statuses: tuple[ApplicationStatus, ...],
        date_from: datetime,
        date_to: datetime,
        interval: TrendInterval,
    ) -> TrendSeries:
        rows = await self.repository.status_transition_trend(
            user_id=user_id, to_statuses=to_statuses, date_from=date_from, date_to=date_to, interval=interval
        )
        return TrendSeries(
            interval=interval, points=_fill_trend(rows, date_from=date_from, date_to=date_to, interval=interval)
        )

    async def _acceptance_rate(
        self, user_id: uuid.UUID, date_from: datetime, date_to: datetime, interval: TrendInterval
    ) -> AcceptanceRateSeries:
        rows = await self.repository.acceptance_rate_rows(
            user_id=user_id, date_from=date_from, date_to=date_to, interval=interval
        )
        per_bucket: dict[datetime, dict[str, int]] = {}
        for bucket, to_status, count in rows:
            truncated = _truncate(bucket, interval)
            entry = per_bucket.setdefault(truncated, {"offers": 0, "accepted": 0})
            if to_status == ApplicationStatus.OFFER:
                entry["offers"] += count
            elif to_status == ApplicationStatus.ACCEPTED:
                entry["accepted"] += count

        points = []
        for bucket in _bucket_starts(date_from, date_to, interval):
            entry = per_bucket.get(bucket, {"offers": 0, "accepted": 0})
            rate = round(entry["accepted"] / entry["offers"], 4) if entry["offers"] else 0.0
            points.append(
                AcceptanceRatePoint(
                    bucket=bucket, offers=entry["offers"], accepted=entry["accepted"], acceptance_rate=rate
                )
            )

        overall_offers = sum(p.offers for p in points)
        overall_accepted = sum(p.accepted for p in points)
        return AcceptanceRateSeries(
            interval=interval,
            points=points,
            overall_offers=overall_offers,
            overall_accepted=overall_accepted,
            overall_acceptance_rate=(
                round(overall_accepted / overall_offers, 4) if overall_offers else 0.0
            ),
        )

    async def _resume_performance(
        self, user_id: uuid.UUID, date_from: datetime, date_to: datetime, interval: TrendInterval
    ) -> ResumePerformance:
        total_resumes, total_versions, parsed_versions = await self.repository.resume_counts(
            user_id=user_id
        )
        average_skills = await self.repository.average_skills_per_version(user_id=user_id)
        trend_rows = await self.repository.resume_versions_trend(
            user_id=user_id, date_from=date_from, date_to=date_to, interval=interval
        )
        return ResumePerformance(
            total_resumes=total_resumes,
            total_versions=total_versions,
            parsed_versions=parsed_versions,
            parse_success_rate=round(parsed_versions / total_versions, 4) if total_versions else 0.0,
            average_skills_per_version=average_skills,
            versions_trend=_fill_trend(trend_rows, date_from=date_from, date_to=date_to, interval=interval),
        )

    async def _ats_resume_performance(
        self, user_id: uuid.UUID, date_from: datetime, date_to: datetime, interval: TrendInterval
    ) -> ATSResumePerformance:
        total_ats_resumes, total_versions, completed_versions = await self.repository.ats_resume_counts(
            user_id=user_id
        )
        trend_rows = await self.repository.ats_score_trend(
            user_id=user_id, date_from=date_from, date_to=date_to, interval=interval
        )
        trend_by_bucket = {
            _truncate(bucket, interval): (avg_ats, avg_kw, n) for bucket, avg_ats, avg_kw, n in trend_rows
        }
        score_trend = [
            ATSScoreTrendPoint(
                bucket=bucket,
                average_ats_score=trend_by_bucket.get(bucket, (None, None, 0))[0],
                average_keyword_score=trend_by_bucket.get(bucket, (None, None, 0))[1],
                sample_size=trend_by_bucket.get(bucket, (None, None, 0))[2],
            )
            for bucket in _bucket_starts(date_from, date_to, interval)
        ]

        scores = await self.repository.ats_score_values(user_id=user_id)
        distribution = [
            ScoreDistributionBucket(
                range_label=label, count=sum(1 for s in scores if low <= s <= high)
            )
            for label, low, high in _SCORE_BANDS
        ]
        average_ats_score = round(sum(scores) / len(scores), 2) if scores else None

        all_keyword_avgs = [avg_kw for _, _, avg_kw, n in trend_rows if avg_kw is not None and n]
        average_keyword_score = (
            round(sum(all_keyword_avgs) / len(all_keyword_avgs), 2) if all_keyword_avgs else None
        )

        return ATSResumePerformance(
            total_ats_resumes=total_ats_resumes,
            total_versions=total_versions,
            completed_versions=completed_versions,
            completion_rate=round(completed_versions / total_versions, 4) if total_versions else 0.0,
            average_ats_score=average_ats_score,
            average_keyword_score=average_keyword_score,
            score_trend=score_trend,
            score_distribution=distribution,
        )

    async def _match_summary(
        self, user_id: uuid.UUID, limit: int
    ) -> tuple[float | None, list[MatchedCompanyCount]]:
        """`average_match_score` + `top_matched_companies`, both derived
        from `MatchingRepository.list_matches_for_user` (already-computed
        `ResumeJobMatch` rows) — no new scoring logic, just a report-shaped
        aggregation over data `matching` already produced.
        """
        matches = await self.matching_repository.list_matches_for_user(
            user_id=user_id, limit=self._settings.MATCHING_CANDIDATE_LIMIT
        )
        scored = [m for m in matches if m.match_score is not None]
        if not scored:
            return None, []

        average_match_score = round(sum(float(m.match_score) for m in scored) / len(scored), 2)

        jobs_by_id = await self.matching_repository.get_jobs_by_ids([m.job_id for m in scored])
        scores_by_company: dict[str, list[float]] = {}
        for match in scored:
            job = jobs_by_id.get(match.job_id)
            if job is None:
                continue
            scores_by_company.setdefault(job.company.name, []).append(float(match.match_score))

        ranked = sorted(
            scores_by_company.items(), key=lambda kv: sum(kv[1]) / len(kv[1]), reverse=True
        )
        top_matched_companies = [
            MatchedCompanyCount(
                company_name=name,
                matches=len(scores),
                average_match_score=round(sum(scores) / len(scores), 2),
            )
            for name, scores in ranked[:limit]
        ]
        return average_match_score, top_matched_companies

    async def _top_skills(self, user_id: uuid.UUID, limit: int) -> list[SkillCount]:
        skill_lists = await self.repository.current_resume_skill_lists(user_id=user_id)
        counter, display_names = _count_skills(skill_lists)
        top = counter.most_common(limit)
        return [SkillCount(skill=display_names[key], count=count) for key, count in top]

    async def _top_companies(
        self, user_id: uuid.UUID, date_from: datetime, date_to: datetime, limit: int
    ) -> list[CompanyOutcome]:
        rows = await self.repository.company_outcomes(
            user_id=user_id, date_from=date_from, date_to=date_to, limit=limit
        )
        return [
            CompanyOutcome(company_name=name, applications=apps, interviews=interviews, offers=offers)
            for name, apps, interviews, offers in rows
        ]

    async def _top_countries(
        self, user_id: uuid.UUID, date_from: datetime, date_to: datetime, limit: int
    ) -> list[CountryCount]:
        rows = await self.repository.application_locations(
            user_id=user_id, date_from=date_from, date_to=date_to
        )
        counter: Counter[str] = Counter()
        for location, job_country in rows:
            country = job_country or _guess_country(location)
            counter[country] += 1
        return [CountryCount(country=c, count=n) for c, n in counter.most_common(limit)]

    async def _hiring_trends(
        self, date_from: datetime, date_to: datetime, interval: TrendInterval, limit: int
    ) -> HiringTrends:
        trend_rows = await self.repository.job_posting_trend(
            date_from=date_from, date_to=date_to, interval=interval
        )
        top_companies_rows = await self.repository.top_hiring_companies(
            date_from=date_from, date_to=date_to, limit=limit
        )
        return HiringTrends(
            interval=interval,
            trend=[
                HiringTrendPoint(bucket=p.bucket, jobs_posted=p.count)
                for p in _fill_trend(trend_rows, date_from=date_from, date_to=date_to, interval=interval)
            ],
            top_hiring_companies=[
                HiringCompanyCount(company_name=name, jobs_posted=count) for name, count in top_companies_rows
            ],
        )

    async def _skill_demand(self, user_id: uuid.UUID, limit: int) -> list[SkillDemandItem]:
        demand_rows = await self.repository.skill_demand(limit=limit)
        skill_lists = await self.repository.current_resume_skill_lists(user_id=user_id)
        user_skills = {skill.strip().lower() for skills in skill_lists for skill in skills if skill}
        return [
            SkillDemandItem(skill=name, job_count=count, user_has_skill=name.strip().lower() in user_skills)
            for name, count in demand_rows
        ]

    async def _response_time(
        self, user_id: uuid.UUID, date_from: datetime, date_to: datetime, interval: TrendInterval
    ) -> ResponseTimeSeries:
        rows = await self.repository.response_time_series(
            user_id=user_id, date_from=date_from, date_to=date_to, interval=interval
        )
        by_bucket = {_truncate(b, interval): (avg, n) for b, avg, n in rows}
        points = [
            ResponseTimePoint(
                bucket=bucket, average_days=by_bucket.get(bucket, (None, 0))[0], sample_size=by_bucket.get(bucket, (None, 0))[1]
            )
            for bucket in _bucket_starts(date_from, date_to, interval)
        ]
        overall = await self.repository.overall_average_response_days(user_id=user_id)
        return ResponseTimeSeries(interval=interval, overall_average_days=overall, points=points)

    async def _heatmap(
        self, user_id: uuid.UUID, date_from: datetime, date_to: datetime
    ) -> ApplicationHeatmap:
        window_from = max(date_from, date_to - timedelta(days=_HEATMAP_WINDOW_DAYS))
        rows = await self.repository.application_counts_by_date(
            user_id=user_id, date_from=window_from, date_to=date_to
        )
        counts = {day: count for day, count in rows}

        cells: list[HeatmapCell] = []
        current = window_from.date()
        end = date_to.date()
        while current <= end:
            cells.append(HeatmapCell(date=current, count=counts.get(current, 0)))
            current += timedelta(days=1)

        total = sum(c.count for c in cells)
        longest = current_streak = running = 0
        for cell in cells:
            if cell.count > 0:
                running += 1
                longest = max(longest, running)
            else:
                running = 0
        for cell in reversed(cells):
            if cell.count > 0:
                current_streak += 1
            else:
                break

        return ApplicationHeatmap(
            cells=cells, total=total, current_streak_days=current_streak, longest_streak_days=longest
        )

    # --- exports -------------------------------------------------------------------------

    async def export_applications_csv(self, *, user_id: uuid.UUID, filters: AnalyticsFilters) -> str:
        applications = await self.repository.list_for_export(user_id=user_id, filters=filters)
        rows = [ExportApplicationRow.model_validate(a) for a in applications]
        return generate_applications_csv(rows)

    async def export_dashboard_pdf(self, *, user_id: uuid.UUID, filters: AnalyticsFilters) -> bytes:
        dashboard = await self.get_dashboard(user_id=user_id, filters=filters)
        date_from, date_to = self._resolve_window(filters)
        content = AnalyticsReportContent(
            title="Job Search Analytics Report",
            generated_at=dashboard.generated_at.strftime("%Y-%m-%d %H:%M UTC"),
            date_range_label=f"{date_from.date().isoformat()} to {date_to.date().isoformat()}",
            summary_rows=[
                ("Total Applications", str(dashboard.summary.total_applications)),
                ("Interview Rate", f"{dashboard.summary.interview_rate:.0%}"),
                ("Offer Rate", f"{dashboard.summary.offer_rate:.0%}"),
                ("Acceptance Rate", f"{dashboard.summary.acceptance_rate:.0%}"),
                (
                    "Avg. Response Time (days)",
                    f"{dashboard.summary.average_response_days:.1f}"
                    if dashboard.summary.average_response_days is not None
                    else "N/A",
                ),
            ],
            top_companies=[
                (c.company_name, str(c.applications), str(c.interviews), str(c.offers))
                for c in dashboard.top_companies
            ],
            top_skills=[(s.skill, str(s.count)) for s in dashboard.top_skills],
            top_countries=[(c.country, str(c.count)) for c in dashboard.top_countries],
        )
        return generate_analytics_pdf(content)


__all__ = ["AnalyticsService"]
