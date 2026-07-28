"""Search module — natural-language query parsing ("Better Search").

Turns a single free-text query like "Find remote Python jobs paying above
₹20 LPA" into structured `SearchFilters` fields (work_mode, skills, salary,
etc.) plus a `cleaned_query` string carrying the remaining semantic intent
("Python developer") — reusing `AIService.extract_structured()` exactly the
way `job_ingestion/tasks.py::_extract_job_skills` already does (schema +
system prompt -> constrained JSON), so this adds no new AI integration
surface.

Deliberately a *parse* step, not a silent rewrite of the search request: the
router returns `ParsedSearchQuery` on its own endpoint so the frontend can
show the user what was extracted (as removable chips) before/while running
the actual `GET /search` call. A misparsed salary or location filter
silently zeroing out results is worse UX than one extra request.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.modules.job_ingestion.models import EmploymentType, SalaryPeriod, WorkMode
from app.modules.search.schemas import ExperienceLevel
from app.services.ai.service import AIService

NL_QUERY_SYSTEM_PROMPT = (
    "You extract structured job-search filters from a single free-text query typed by a "
    "job seeker. Only extract a field when the query explicitly states it — never infer or "
    "guess a filter that isn't clearly present. Leave every field you're not confident about "
    "as null/empty.\n\n"
    "- work_mode: one of remote, hybrid, onsite — only if explicitly mentioned.\n"
    "- skills: concrete technical/professional skills or job-role keywords named in the "
    "query (e.g. 'Python', 'FastAPI', 'product manager').\n"
    "- salary_min / salary_max: numeric salary bound if the query states one (e.g. 'above 20 "
    "LPA' -> salary_min=2000000, salary_currency='INR', salary_period='yearly'; '$120k+' -> "
    "salary_min=120000, salary_currency='USD'). Resolve LPA/lakh/k shorthand to the full "
    "number in the stated currency's smallest whole unit (LPA/lakh values are in INR).\n"
    "- salary_currency: 3-letter ISO code if a salary was extracted, else null.\n"
    "- salary_period: 'yearly', 'monthly', or 'hourly' if a salary was extracted and a period "
    "is stated or clearly implied (LPA/lakh implies yearly), else null.\n"
    "- experience_level: one of internship, entry, mid, senior, lead, executive — only if "
    "explicitly stated (e.g. 'senior', 'entry level').\n"
    "- employment_type: one of full_time, part_time, contract, internship — only if stated.\n"
    "- country / city: location names explicitly mentioned (not 'remote', which is a "
    "work_mode, not a location).\n"
    "- cleaned_query: the remaining free-text search intent after removing the filter phrases "
    "you extracted above (e.g. from 'Find remote Python jobs paying above 20 LPA' with "
    "work_mode=remote and salary_min extracted, cleaned_query should be 'Python developer "
    "jobs' or similar — keep it natural, not a keyword dump)."
)


class ParsedSearchQuery(BaseModel):
    """Structured filters extracted from one natural-language query, plus
    the semantic remainder. Every filter field is optional/`None` when the
    query gave no signal for it — the caller applies only what's present,
    it never fabricates a default.
    """

    work_mode: WorkMode | None = None
    skills: list[str] = Field(default_factory=list)
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    salary_period: SalaryPeriod | None = None
    experience_level: ExperienceLevel | None = None
    employment_type: EmploymentType | None = None
    country: str | None = None
    city: str | None = None
    cleaned_query: str


class _NLQueryExtraction(BaseModel):
    """Loose extraction schema for the LLM call — string enums so a
    slightly-off model output ("Remote" vs "remote") doesn't hard-fail
    validation; normalized into `ParsedSearchQuery`'s real enums afterward.
    """

    work_mode: str | None = None
    skills: list[str] = Field(default_factory=list)
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    salary_period: str | None = None
    experience_level: str | None = None
    employment_type: str | None = None
    country: str | None = None
    city: str | None = None
    cleaned_query: str = ""


def _normalize_enum(value: str | None, enum_cls: type) -> object | None:
    if not value:
        return None
    try:
        return enum_cls(value.strip().lower())
    except ValueError:
        return None


async def parse_natural_language_query(
    raw_query: str, *, ai_service: AIService
) -> ParsedSearchQuery:
    """Parse one free-text query into structured filters. Falls back to an
    all-empty-filters result (cleaned_query = the original text unchanged)
    if the AI provider is unavailable or returns something unusable — the
    caller then simply runs the original query as plain keyword/semantic
    search, same as before this feature existed.
    """
    if not raw_query or not raw_query.strip():
        return ParsedSearchQuery(cleaned_query=raw_query or "")

    try:
        extracted = await ai_service.extract_structured(
            text=raw_query,
            schema=_NLQueryExtraction.model_json_schema(),
            system=NL_QUERY_SYSTEM_PROMPT,
            max_tokens=600,
        )
        parsed = _NLQueryExtraction.model_validate(extracted)
    except Exception:
        return ParsedSearchQuery(cleaned_query=raw_query)

    return ParsedSearchQuery(
        work_mode=_normalize_enum(parsed.work_mode, WorkMode),
        skills=parsed.skills,
        salary_min=parsed.salary_min,
        salary_max=parsed.salary_max,
        salary_currency=(parsed.salary_currency or "").upper()[:3] or None,
        salary_period=_normalize_enum(parsed.salary_period, SalaryPeriod),
        experience_level=_normalize_enum(parsed.experience_level, ExperienceLevel),
        employment_type=_normalize_enum(parsed.employment_type, EmploymentType),
        country=parsed.country or None,
        city=parsed.city or None,
        cleaned_query=parsed.cleaned_query.strip() or raw_query,
    )


__all__ = ["ParsedSearchQuery", "parse_natural_language_query", "NL_QUERY_SYSTEM_PROMPT"]
