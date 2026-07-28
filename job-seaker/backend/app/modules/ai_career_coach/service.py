"""AI Career Coach module — application/use-case layer.

Each analysis feature follows the same shape: gather grounding context
(parsed resume data and/or real `job_ingestion` aggregates), call
`AIService` for the parts that genuinely need judgment/generation, compute
anything that can be computed deterministically in Python rather than
trusted to the model (readiness scores, skill-match percentages, salary
stats), and persist the result as a `CareerInsight` so history is free to
re-read. The chat assistant (`send_message`) is the one free-form,
un-cached feature — see `CoachConversation`/`CoachMessage`.
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.exceptions import ConflictError, NotFoundError
from app.core.logging import get_logger
from app.modules.ai_career_coach.models import (
    CareerInsight,
    CareerInsightType,
    CoachConversation,
    CoachMessage,
    CoachMessageRole,
)
from app.modules.ai_career_coach.repository import (
    CareerInsightRepository,
    CoachConversationRepository,
    CoachMessageRepository,
    JobMarketRepository,
    ResumeContextRepository,
)
from app.modules.ai_career_coach.schemas import (
    CareerRoadmapResult,
    CareerSuggestionResult,
    CompanyDemand,
    CountryDemand,
    CountryDemandResult,
    InterviewPrepResult,
    JobReadinessScoreResult,
    LearningRecommendationResult,
    MarketTrendResult,
    ResumeReviewResult,
    SalaryPredictionResult,
    SkillDemand,
    SkillGapItem,
    SkillGapResult,
)
from app.modules.resume_parser.models import ResumeVersion, ResumeVersionStatus
from app.services.ai.service import AIService

logger = get_logger(__name__)


# --- local, non-public extraction shapes ---------------------------------------------------
# Narrower than the public result schemas above: these back single
# structured-extraction calls whose remaining fields are filled in
# deterministically in Python rather than trusted to the model.


class _SkillGapMissingOnly(BaseModel):
    missing_skills: list[SkillGapItem] = Field(default_factory=list)


class _SalaryEstimate(BaseModel):
    predicted_min: float
    predicted_max: float
    currency: str
    narrative: str


class _ReadinessNarrative(BaseModel):
    narrative: str
    recommendations: list[str] = Field(default_factory=list)


def _resume_context_text(version: ResumeVersion) -> str:
    """Render a resume's parsed data as plain text for prompting."""
    parsed = version.parsed_data
    if parsed is None:
        return ""

    lines: list[str] = []
    if parsed.summary:
        lines.append(f"Summary: {parsed.summary}")
    if parsed.skills:
        lines.append("Skills: " + ", ".join(parsed.skills))
    for exp in parsed.experience:
        span = f"{exp.get('start_date') or '?'} - {exp.get('end_date') or 'Present'}"
        lines.append(
            f"Experience: {exp.get('title', '')} at {exp.get('company', '')} ({span}): "
            f"{exp.get('description') or ''}"
        )
        for highlight in exp.get("highlights") or []:
            lines.append(f"  - {highlight}")
    for edu in parsed.education:
        lines.append(
            f"Education: {edu.get('degree', '')} in {edu.get('field_of_study', '')} "
            f"from {edu.get('institution', '')}"
        )
    for project in parsed.projects:
        lines.append(f"Project: {project.get('name', '')} — {project.get('description') or ''}")
    if parsed.certifications:
        lines.append(
            "Certifications: " + ", ".join(c.get("name", "") for c in parsed.certifications)
        )
    if parsed.achievements:
        lines.append("Achievements: " + "; ".join(parsed.achievements))
    return "\n".join(lines)


def _truncate_title(message: str, max_length: int = 80) -> str:
    stripped = message.strip().replace("\n", " ")
    if len(stripped) <= max_length:
        return stripped or "New conversation"
    return stripped[: max_length - 1].rstrip() + "…"


_CHAT_SYSTEM_PROMPT = (
    "You are an AI career coach: warm, direct, and practical. Give specific, actionable advice "
    "about resumes, job searching, interviews, career growth, and the job market. Keep replies "
    "concise (a few short paragraphs or a short list) unless the user asks for depth. If you "
    "don't have enough information to give specific advice, ask a clarifying question."
)


class AICareerCoachService:
    """Business logic orchestration for the AI Career Coach module."""

    def __init__(
        self,
        insight_repository: CareerInsightRepository,
        conversation_repository: CoachConversationRepository,
        message_repository: CoachMessageRepository,
        resume_context_repository: ResumeContextRepository,
        job_market_repository: JobMarketRepository,
        ai_service: AIService,
    ) -> None:
        self.insight_repository = insight_repository
        self.conversation_repository = conversation_repository
        self.message_repository = message_repository
        self.resume_context_repository = resume_context_repository
        self.job_market_repository = job_market_repository
        self.ai_service = ai_service
        self.settings = get_settings()

    # --- shared helpers --------------------------------------------------------------------

    async def _require_parsed_resume_version(
        self, *, user_id: uuid.UUID, resume_version_id: uuid.UUID
    ) -> ResumeVersion:
        version = await self.resume_context_repository.get_owned_version(
            user_id=user_id, resume_version_id=resume_version_id
        )
        if version is None:
            raise NotFoundError("Resume version not found.")
        if version.status != ResumeVersionStatus.PARSED or version.parsed_data is None:
            raise ConflictError(
                "Resume must finish parsing before requesting career coach analysis.",
                error_code="resume_not_parsed",
            )
        return version

    async def _save_insight(
        self,
        *,
        user_id: uuid.UUID,
        insight_type: CareerInsightType,
        resume_version_id: uuid.UUID | None,
        input_params: dict,
        result: dict,
    ) -> CareerInsight:
        return await self.insight_repository.create(
            user_id=user_id,
            insight_type=insight_type,
            resume_version_id=resume_version_id,
            input_params=input_params,
            result=result,
            model_used=self.ai_service.model_name,
        )

    # --- resume review + improvement suggestions --------------------------------------------

    async def review_resume(
        self, *, user_id: uuid.UUID, resume_version_id: uuid.UUID
    ) -> ResumeReviewResult:
        version = await self._require_parsed_resume_version(
            user_id=user_id, resume_version_id=resume_version_id
        )
        extracted = await self.ai_service.extract_structured(
            text=_resume_context_text(version),
            schema=ResumeReviewResult.model_json_schema(),
            system=(
                "You are an expert resume reviewer and career coach. Review the resume content "
                "below and give honest, specific, actionable feedback, including concrete "
                "rewrite suggestions per section. Base every observation strictly on the "
                "content given — do not invent experience, skills, or employers."
            ),
            max_tokens=self.settings.AI_CAREER_COACH_ANALYSIS_MAX_TOKENS,
        )
        result = ResumeReviewResult.model_validate(extracted)
        await self._save_insight(
            user_id=user_id,
            insight_type=CareerInsightType.RESUME_REVIEW,
            resume_version_id=resume_version_id,
            input_params={},
            result=result.model_dump(mode="json"),
        )
        return result

    async def _get_or_create_resume_review(
        self, *, user_id: uuid.UUID, resume_version_id: uuid.UUID
    ) -> ResumeReviewResult:
        cached = await self.insight_repository.get_latest(
            user_id=user_id,
            insight_type=CareerInsightType.RESUME_REVIEW,
            resume_version_id=resume_version_id,
        )
        if cached is not None:
            return ResumeReviewResult.model_validate(cached.result)
        return await self.review_resume(user_id=user_id, resume_version_id=resume_version_id)

    # --- career suggestions -----------------------------------------------------------------

    async def suggest_careers(
        self, *, user_id: uuid.UUID, resume_version_id: uuid.UUID
    ) -> CareerSuggestionResult:
        version = await self._require_parsed_resume_version(
            user_id=user_id, resume_version_id=resume_version_id
        )
        extracted = await self.ai_service.extract_structured(
            text=_resume_context_text(version),
            schema=CareerSuggestionResult.model_json_schema(),
            system=(
                "You are a career coach. Based on this candidate's resume, suggest 3-6 job "
                "titles or career paths that fit their skills and experience, including "
                "adjacent or growth roles they may not have considered. Ground every "
                "suggestion in specifics from the resume."
            ),
            max_tokens=self.settings.AI_CAREER_COACH_ANALYSIS_MAX_TOKENS,
        )
        result = CareerSuggestionResult.model_validate(extracted)
        await self._save_insight(
            user_id=user_id,
            insight_type=CareerInsightType.CAREER_SUGGESTIONS,
            resume_version_id=resume_version_id,
            input_params={},
            result=result.model_dump(mode="json"),
        )
        return result

    # --- skill gap analysis -----------------------------------------------------------------

    async def analyze_skill_gap(
        self, *, user_id: uuid.UUID, resume_version_id: uuid.UUID, target_role: str
    ) -> SkillGapResult:
        version = await self._require_parsed_resume_version(
            user_id=user_id, resume_version_id=resume_version_id
        )
        resume_skills = version.parsed_data.skills or []
        resume_skills_normalized = {s.strip().lower() for s in resume_skills if s.strip()}

        required = await self.job_market_repository.required_skills_for_title(
            title_query=target_role,
            limit=self.settings.AI_CAREER_COACH_SKILL_GAP_REQUIRED_SKILL_LIMIT,
        )
        grounded = bool(required)

        if grounded:
            required_names = [name for name, _count in required]
            matched_skills = [
                name for name in required_names if name.strip().lower() in resume_skills_normalized
            ]
            candidate_missing = [
                name
                for name in required_names
                if name.strip().lower() not in resume_skills_normalized
            ]
            overall_readiness_percent = round(
                (len(matched_skills) / len(required_names)) * 100, 1
            )
            skills_hint = (
                "The following skills are commonly required in real job postings for this "
                "role and are NOT present on the candidate's resume: "
                + ", ".join(candidate_missing)
                + ". For each one, explain why it matters for this role and how important it is."
                if candidate_missing
                else "The candidate already has every commonly-required skill found in real "
                "postings for this role. Return an empty missing_skills list."
            )
        else:
            matched_skills = resume_skills
            candidate_missing = []
            overall_readiness_percent = 0.0  # recomputed below once the model responds
            skills_hint = (
                f"No live job posting data is available for '{target_role}'. Based on general "
                "knowledge of this role, identify the skills most commonly required that are "
                "missing from the candidate's resume."
            )

        prompt = f"{_resume_context_text(version)}\n\nTarget role: {target_role}\n\n{skills_hint}"
        extracted = await self.ai_service.extract_structured(
            text=prompt,
            schema=_SkillGapMissingOnly.model_json_schema(),
            system=(
                "You are a career coach analyzing a skill gap for a candidate targeting a "
                "specific role. Only report skills genuinely relevant to the target role."
            ),
            max_tokens=self.settings.AI_CAREER_COACH_ANALYSIS_MAX_TOKENS,
        )
        missing_skills = _SkillGapMissingOnly.model_validate(extracted).missing_skills

        if grounded:
            allowed = {name.strip().lower() for name in candidate_missing}
            missing_skills = [item for item in missing_skills if item.skill.strip().lower() in allowed]
        else:
            overall_readiness_percent = max(0.0, 100.0 - len(missing_skills) * 10.0)

        result = SkillGapResult(
            target_role=target_role,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            overall_readiness_percent=overall_readiness_percent,
        )
        await self._save_insight(
            user_id=user_id,
            insight_type=CareerInsightType.SKILL_GAP_ANALYSIS,
            resume_version_id=resume_version_id,
            input_params={"target_role": target_role},
            result=result.model_dump(mode="json"),
        )
        return result

    # --- learning recommendations -----------------------------------------------------------

    async def recommend_learning(
        self,
        *,
        user_id: uuid.UUID,
        resume_version_id: uuid.UUID,
        target_role: str,
        skills: list[str] | None = None,
    ) -> LearningRecommendationResult:
        version = await self._require_parsed_resume_version(
            user_id=user_id, resume_version_id=resume_version_id
        )

        if skills:
            target_skills = skills
        else:
            gap = await self.analyze_skill_gap(
                user_id=user_id, resume_version_id=resume_version_id, target_role=target_role
            )
            target_skills = [item.skill for item in gap.missing_skills]

        if not target_skills:
            result = LearningRecommendationResult(target_role=target_role, recommendations=[])
        else:
            prompt = (
                f"{_resume_context_text(version)}\n\nTarget role: {target_role}\n"
                f"Skills to develop: {', '.join(target_skills)}"
            )
            extracted = await self.ai_service.extract_structured(
                text=prompt,
                schema=LearningRecommendationResult.model_json_schema(),
                system=(
                    "You are a career coach recommending learning resources. For each skill, "
                    "suggest one concrete, well-known resource type (course, certification, "
                    "book, practice project) — general provider categories are fine (e.g. "
                    "'Coursera', 'official docs') when you don't know a specific title."
                ),
                max_tokens=self.settings.AI_CAREER_COACH_ANALYSIS_MAX_TOKENS,
            )
            result = LearningRecommendationResult.model_validate(extracted)
            result.target_role = target_role

        await self._save_insight(
            user_id=user_id,
            insight_type=CareerInsightType.LEARNING_RECOMMENDATIONS,
            resume_version_id=resume_version_id,
            input_params={"target_role": target_role, "skills": skills},
            result=result.model_dump(mode="json"),
        )
        return result

    # --- salary prediction -------------------------------------------------------------------

    async def predict_salary(
        self,
        *,
        user_id: uuid.UUID,
        target_role: str,
        country: str | None = None,
        years_experience: float | None = None,
        resume_version_id: uuid.UUID | None = None,
    ) -> SalaryPredictionResult:
        stats = await self.job_market_repository.salary_stats_for_role(
            title_query=target_role,
            country=country,
            sample_limit=self.settings.AI_CAREER_COACH_SALARY_SAMPLE_LIMIT,
        )
        grounded = (
            stats["sample_size"] >= self.settings.AI_CAREER_COACH_SALARY_MIN_SAMPLE_SIZE
            and stats["avg_min"] is not None
            and stats["avg_max"] is not None
        )

        context_lines = [f"Target role: {target_role}"]
        if country:
            context_lines.append(f"Country: {country}")
        if years_experience is not None:
            context_lines.append(f"Years of experience: {years_experience}")
        if resume_version_id is not None:
            version = await self.resume_context_repository.get_owned_version(
                user_id=user_id, resume_version_id=resume_version_id
            )
            if version is not None and version.parsed_data is not None:
                context_lines.append(_resume_context_text(version))

        if grounded:
            predicted_min = round(stats["avg_min"], 2)
            predicted_max = round(stats["avg_max"], 2)
            currency = stats["currency"] or "USD"
            context_lines.append(
                f"Real job posting data ({stats['sample_size']} postings) shows an average "
                f"salary range of {predicted_min}-{predicted_max} {currency} for this role."
            )
            narrative = await self.ai_service.complete_chat(
                system=(
                    "You are a career coach. In 2-3 sentences, explain this salary range to "
                    "the candidate in plain language, referencing their experience level if "
                    "given."
                ),
                messages=[{"role": "user", "content": "\n".join(context_lines)}],
                max_tokens=self.settings.AI_CAREER_COACH_NARRATIVE_MAX_TOKENS,
            )
        else:
            context_lines.append(
                "No sufficient real job posting data is available for this role/location. "
                "Estimate a realistic salary range from general market knowledge."
            )
            extracted = await self.ai_service.extract_structured(
                text="\n".join(context_lines),
                schema=_SalaryEstimate.model_json_schema(),
                system="You are a compensation analyst estimating a realistic salary range.",
                max_tokens=self.settings.AI_CAREER_COACH_ANALYSIS_MAX_TOKENS,
            )
            estimate = _SalaryEstimate.model_validate(extracted)
            predicted_min, predicted_max, currency, narrative = (
                estimate.predicted_min,
                estimate.predicted_max,
                estimate.currency,
                estimate.narrative,
            )

        result = SalaryPredictionResult(
            target_role=target_role,
            country=country,
            predicted_min=predicted_min,
            predicted_max=predicted_max,
            currency=currency,
            based_on_market_data=grounded,
            sample_size=stats["sample_size"],
            narrative=narrative.strip(),
        )
        await self._save_insight(
            user_id=user_id,
            insight_type=CareerInsightType.SALARY_PREDICTION,
            resume_version_id=resume_version_id,
            input_params={
                "target_role": target_role,
                "country": country,
                "years_experience": years_experience,
            },
            result=result.model_dump(mode="json"),
        )
        return result

    # --- career roadmap ----------------------------------------------------------------------

    async def generate_roadmap(
        self,
        *,
        user_id: uuid.UUID,
        resume_version_id: uuid.UUID,
        target_role: str,
        timeframe_months: int,
    ) -> CareerRoadmapResult:
        version = await self._require_parsed_resume_version(
            user_id=user_id, resume_version_id=resume_version_id
        )
        prompt = (
            f"{_resume_context_text(version)}\n\nTarget role: {target_role}\n"
            f"Timeframe: {timeframe_months} months"
        )
        extracted = await self.ai_service.extract_structured(
            text=prompt,
            schema=CareerRoadmapResult.model_json_schema(),
            system=(
                "You are a career coach building a phased roadmap toward a target role. Break "
                f"the {timeframe_months}-month timeframe into 3-5 sequential phases, each with "
                "concrete goals, skills to develop, and suggested actions grounded in the "
                "candidate's actual background."
            ),
            max_tokens=self.settings.AI_CAREER_COACH_ANALYSIS_MAX_TOKENS,
        )
        result = CareerRoadmapResult.model_validate(extracted)
        result.target_role = target_role
        result.timeframe_months = timeframe_months
        await self._save_insight(
            user_id=user_id,
            insight_type=CareerInsightType.CAREER_ROADMAP,
            resume_version_id=resume_version_id,
            input_params={"target_role": target_role, "timeframe_months": timeframe_months},
            result=result.model_dump(mode="json"),
        )
        return result

    # --- interview preparation ---------------------------------------------------------------

    async def prepare_interview(
        self, *, user_id: uuid.UUID, resume_version_id: uuid.UUID, target_role: str
    ) -> InterviewPrepResult:
        version = await self._require_parsed_resume_version(
            user_id=user_id, resume_version_id=resume_version_id
        )
        prompt = f"{_resume_context_text(version)}\n\nTarget role: {target_role}"
        extracted = await self.ai_service.extract_structured(
            text=prompt,
            schema=InterviewPrepResult.model_json_schema(),
            system=(
                "You are an interview coach. Generate a realistic mix of behavioral, "
                "technical, and situational interview questions this candidate is likely to "
                "face for the target role, tailored to their actual background. Include a "
                "short tip for answering each, plus general interview tips."
            ),
            max_tokens=self.settings.AI_CAREER_COACH_ANALYSIS_MAX_TOKENS,
        )
        result = InterviewPrepResult.model_validate(extracted)
        result.target_role = target_role
        await self._save_insight(
            user_id=user_id,
            insight_type=CareerInsightType.INTERVIEW_PREPARATION,
            resume_version_id=resume_version_id,
            input_params={"target_role": target_role},
            result=result.model_dump(mode="json"),
        )
        return result

    # --- job readiness score -----------------------------------------------------------------

    async def compute_job_readiness(
        self, *, user_id: uuid.UUID, resume_version_id: uuid.UUID, target_role: str | None = None
    ) -> JobReadinessScoreResult:
        version = await self._require_parsed_resume_version(
            user_id=user_id, resume_version_id=resume_version_id
        )
        parsed = version.parsed_data

        review = await self._get_or_create_resume_review(
            user_id=user_id, resume_version_id=resume_version_id
        )
        resume_quality_score = review.overall_score

        if target_role:
            gap = await self.analyze_skill_gap(
                user_id=user_id, resume_version_id=resume_version_id, target_role=target_role
            )
            skill_match_score = gap.overall_readiness_percent
        else:
            skill_match_score = min(100.0, len(parsed.skills or []) * 8.0)

        sections_present = [
            bool(parsed.summary and parsed.summary.strip()),
            len(parsed.skills or []) >= 3,
            len(parsed.experience or []) >= 1,
            len(parsed.education or []) >= 1,
            len(parsed.projects or []) >= 1,
            bool(parsed.certifications) or bool(parsed.achievements),
        ]
        completeness_score = round(sum(sections_present) / len(sections_present) * 100, 1)

        settings = self.settings
        overall_score = round(
            resume_quality_score * settings.AI_CAREER_COACH_READINESS_RESUME_WEIGHT
            + skill_match_score * settings.AI_CAREER_COACH_READINESS_SKILL_WEIGHT
            + completeness_score * settings.AI_CAREER_COACH_READINESS_COMPLETENESS_WEIGHT,
            1,
        )

        narrative_prompt = (
            f"Resume quality score: {resume_quality_score}/100\n"
            f"Skill match score: {skill_match_score}/100 "
            f"({'for target role ' + target_role if target_role else 'general, no target role given'})\n"
            f"Profile completeness score: {completeness_score}/100\n"
            f"Key weaknesses noted in resume review: {'; '.join(review.weaknesses) or 'none'}"
        )
        extracted = await self.ai_service.extract_structured(
            text=narrative_prompt,
            schema=_ReadinessNarrative.model_json_schema(),
            system=(
                "You are a career coach. Given these job-readiness sub-scores, write a short "
                "narrative assessment plus 3-5 concrete, prioritized recommendations to "
                "improve job readiness."
            ),
            max_tokens=self.settings.AI_CAREER_COACH_NARRATIVE_MAX_TOKENS,
        )
        narrative_result = _ReadinessNarrative.model_validate(extracted)

        result = JobReadinessScoreResult(
            overall_score=overall_score,
            resume_quality_score=resume_quality_score,
            skill_match_score=skill_match_score,
            completeness_score=completeness_score,
            narrative=narrative_result.narrative,
            recommendations=narrative_result.recommendations,
        )
        await self._save_insight(
            user_id=user_id,
            insight_type=CareerInsightType.JOB_READINESS_SCORE,
            resume_version_id=resume_version_id,
            input_params={"target_role": target_role},
            result=result.model_dump(mode="json"),
        )
        return result

    # --- market trend analysis ---------------------------------------------------------------

    async def analyze_market_trends(
        self, *, user_id: uuid.UUID, title: str | None = None, window_days: int | None = None
    ) -> MarketTrendResult:
        window_days = window_days or self.settings.AI_CAREER_COACH_MARKET_TREND_WINDOW_DAYS
        current, previous = await self.job_market_repository.posting_counts_for_window(
            window_days=window_days, title_filter=title
        )
        growth_percent = round((current - previous) / previous * 100, 1) if previous else None
        top_skills = [
            SkillDemand(skill=name, job_count=count)
            for name, count in await self.job_market_repository.top_skills_for_window(
                window_days=window_days, limit=15, title_filter=title
            )
        ]
        top_companies = [
            CompanyDemand(company=name, job_count=count)
            for name, count in await self.job_market_repository.top_companies_for_window(
                window_days=window_days, limit=10, title_filter=title
            )
        ]

        narrative_prompt = (
            f"Job posting volume over the last {window_days} days"
            + (f" for '{title}'" if title else "")
            + f": {current} postings (previous {window_days}-day window: {previous}, growth: "
            + (f"{growth_percent}%" if growth_percent is not None else "n/a")
            + ").\n"
            f"Top in-demand skills: {', '.join(s.skill for s in top_skills) or 'none'}\n"
            f"Top hiring companies: {', '.join(c.company for c in top_companies) or 'none'}"
        )
        narrative = await self.ai_service.complete_chat(
            system=(
                "You are a labor-market analyst. Summarize this hiring trend data in 2-4 "
                "sentences for a job seeker."
            ),
            messages=[{"role": "user", "content": narrative_prompt}],
            max_tokens=self.settings.AI_CAREER_COACH_NARRATIVE_MAX_TOKENS,
        )

        result = MarketTrendResult(
            window_days=window_days,
            title=title,
            total_postings=current,
            previous_window_postings=previous,
            growth_percent=growth_percent,
            top_skills=top_skills,
            top_companies=top_companies,
            narrative=narrative.strip(),
        )
        await self._save_insight(
            user_id=user_id,
            insight_type=CareerInsightType.MARKET_TREND_ANALYSIS,
            resume_version_id=None,
            input_params={"title": title, "window_days": window_days},
            result=result.model_dump(mode="json"),
        )
        return result

    # --- country-based demand -----------------------------------------------------------------

    async def analyze_country_demand(
        self,
        *,
        user_id: uuid.UUID,
        title: str | None = None,
        skill: str | None = None,
        limit: int | None = None,
    ) -> CountryDemandResult:
        limit = limit or self.settings.AI_CAREER_COACH_COUNTRY_DEMAND_LIMIT
        rows = await self.job_market_repository.demand_by_country(
            title_query=title, skill_name=skill, limit=limit
        )
        countries = [CountryDemand(**row) for row in rows]

        if countries:
            focus = " and ".join(
                part for part in [f"role '{title}'" if title else "", f"skill '{skill}'" if skill else ""] if part
            )
            narrative_prompt = (
                f"Job demand by country{(' for ' + focus) if focus else ''}:\n"
                + "\n".join(f"{c.country}: {c.job_count} postings" for c in countries[:10])
            )
        else:
            narrative_prompt = "No matching job postings were found for this query."

        narrative = await self.ai_service.complete_chat(
            system=(
                "You are a labor-market analyst. Summarize where global demand is "
                "concentrated in 2-3 sentences."
            ),
            messages=[{"role": "user", "content": narrative_prompt}],
            max_tokens=self.settings.AI_CAREER_COACH_NARRATIVE_MAX_TOKENS,
        )

        result = CountryDemandResult(
            title=title, skill=skill, countries=countries, narrative=narrative.strip()
        )
        await self._save_insight(
            user_id=user_id,
            insight_type=CareerInsightType.COUNTRY_DEMAND,
            resume_version_id=None,
            input_params={"title": title, "skill": skill},
            result=result.model_dump(mode="json"),
        )
        return result

    # --- insight history ----------------------------------------------------------------------

    async def list_insights(
        self,
        *,
        user_id: uuid.UUID,
        insight_type: CareerInsightType | None,
        offset: int,
        limit: int,
    ) -> tuple[Sequence[CareerInsight], int]:
        insights = await self.insight_repository.list_by_user(
            user_id=user_id, insight_type=insight_type, offset=offset, limit=limit
        )
        total = await self.insight_repository.count_by_user(
            user_id=user_id, insight_type=insight_type
        )
        return insights, total

    async def get_insight(self, *, user_id: uuid.UUID, insight_id: uuid.UUID) -> CareerInsight:
        insight = await self.insight_repository.get_owned(user_id=user_id, insight_id=insight_id)
        if insight is None:
            raise NotFoundError("Insight not found.")
        return insight

    # --- AI chat assistant + conversation history -----------------------------------------------

    async def send_message(
        self, *, user_id: uuid.UUID, conversation_id: uuid.UUID | None, message: str
    ) -> tuple[CoachConversation, CoachMessage]:
        if conversation_id is not None:
            conversation = await self.conversation_repository.get_owned(
                user_id=user_id, conversation_id=conversation_id
            )
            if conversation is None:
                raise NotFoundError("Conversation not found.")
        else:
            conversation = await self.conversation_repository.create(
                user_id=user_id, title=_truncate_title(message)
            )

        await self.message_repository.create(
            conversation_id=conversation.id, role=CoachMessageRole.USER, content=message
        )

        history = await self.message_repository.list_recent_for_conversation(
            conversation_id=conversation.id,
            limit=self.settings.AI_CAREER_COACH_CHAT_MAX_HISTORY_MESSAGES,
        )
        chat_messages = [{"role": str(m.role), "content": m.content} for m in history]

        system = _CHAT_SYSTEM_PROMPT
        profile_version = await self.resume_context_repository.get_current_version_for_user(
            user_id=user_id
        )
        if profile_version is not None and profile_version.parsed_data is not None:
            system += (
                "\n\nUser's current resume context (for personalization only, don't repeat "
                "it verbatim):\n" + _resume_context_text(profile_version)
            )

        reply_text = await self.ai_service.complete_chat(
            system=system,
            messages=chat_messages,
            max_tokens=self.settings.AI_CAREER_COACH_CHAT_MAX_TOKENS,
        )
        assistant_message = await self.message_repository.create(
            conversation_id=conversation.id, role=CoachMessageRole.ASSISTANT, content=reply_text
        )
        await self.conversation_repository.update(
            conversation, last_message_at=datetime.now(timezone.utc)
        )
        return conversation, assistant_message

    async def list_conversations(
        self, *, user_id: uuid.UUID, offset: int, limit: int
    ) -> tuple[Sequence[CoachConversation], int]:
        conversations = await self.conversation_repository.list_by_user(
            user_id=user_id, offset=offset, limit=limit
        )
        total = await self.conversation_repository.count_by_user(user_id=user_id)
        return conversations, total

    async def get_conversation(
        self, *, user_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> CoachConversation:
        conversation = await self.conversation_repository.get_owned(
            user_id=user_id, conversation_id=conversation_id, with_messages=True
        )
        if conversation is None:
            raise NotFoundError("Conversation not found.")
        return conversation

    async def delete_conversation(self, *, user_id: uuid.UUID, conversation_id: uuid.UUID) -> None:
        conversation = await self.conversation_repository.get_owned(
            user_id=user_id, conversation_id=conversation_id
        )
        if conversation is None:
            raise NotFoundError("Conversation not found.")
        await self.conversation_repository.update(
            conversation, deleted_at=datetime.now(timezone.utc)
        )
