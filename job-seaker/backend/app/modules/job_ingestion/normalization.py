"""Job Ingestion module — pure normalization helpers.

Mirrors `resume_parser/extraction.py`: no DB session, no network calls, no
Celery — just deterministic parsing functions over adapter-supplied raw text
so they can be unit tested in isolation. `tasks.py` is the only caller.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.modules.job_ingestion.models import EmploymentType, SalaryPeriod, WorkMode

_CURRENCY_SYMBOLS: dict[str, str] = {"$": "USD", "£": "GBP", "€": "EUR", "₹": "INR"}
_CURRENCY_CODES = ("USD", "GBP", "EUR", "INR", "CAD", "AUD", "SGD")

_NUMBER_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*([kK])?")

_HOURLY_KEYWORDS = ("/hr", "/ hr", "hour", "hourly")
_MONTHLY_KEYWORDS = ("/mo", "/ mo", "month", "monthly")
_YEARLY_KEYWORDS = ("/yr", "/ yr", "year", "annum", "annual")

_EMPLOYMENT_TYPE_KEYWORDS: dict[EmploymentType, tuple[str, ...]] = {
    EmploymentType.INTERNSHIP: ("intern",),
    EmploymentType.TEMPORARY: ("temp",),
    EmploymentType.FREELANCE: ("freelance",),
    EmploymentType.CONTRACT: ("contract", "contractor"),
    EmploymentType.PART_TIME: ("part-time", "part time", "parttime"),
    EmploymentType.FULL_TIME: ("full-time", "full time", "fulltime"),
}

_REMOTE_KEYWORDS = ("remote", "work from home", "wfh")
_HYBRID_KEYWORDS = ("hybrid",)
_ONSITE_KEYWORDS = ("on-site", "onsite", "in office", "in-office")

_COUNTRY_CODE_MAP: dict[str, str] = {
    "united states": "US",
    "united states of america": "US",
    "usa": "US",
    "us": "US",
    "united kingdom": "GB",
    "uk": "GB",
    "canada": "CA",
    "australia": "AU",
    "germany": "DE",
    "france": "FR",
    "india": "IN",
    "ireland": "IE",
    "netherlands": "NL",
    "singapore": "SG",
}


@dataclass(frozen=True)
class SalaryInfo:
    salary_min: float | None
    salary_max: float | None
    currency: str | None
    period: SalaryPeriod


@dataclass(frozen=True)
class LocationParts:
    city: str | None
    region: str | None
    country: str | None
    country_code: str | None
    is_remote: bool


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: list[str]


def _detect_currency(raw: str) -> str | None:
    for symbol, code in _CURRENCY_SYMBOLS.items():
        if symbol in raw:
            return code
    upper = raw.upper()
    for code in _CURRENCY_CODES:
        if re.search(rf"\b{code}\b", upper):
            return code
    return None


def _detect_period(raw_lower: str) -> SalaryPeriod:
    if any(kw in raw_lower for kw in _HOURLY_KEYWORDS):
        return SalaryPeriod.HOURLY
    if any(kw in raw_lower for kw in _MONTHLY_KEYWORDS):
        return SalaryPeriod.MONTHLY
    if any(kw in raw_lower for kw in _YEARLY_KEYWORDS):
        return SalaryPeriod.YEARLY
    return SalaryPeriod.UNKNOWN


def _parse_numbers(raw: str) -> list[float]:
    values: list[float] = []
    for match in _NUMBER_RE.finditer(raw):
        digits, k_suffix = match.group(1), match.group(2)
        value = float(digits.replace(",", ""))
        if k_suffix:
            value *= 1000
        values.append(value)
    return values


def parse_salary(raw: str | None) -> SalaryInfo:
    """Best-effort parse of a free-text salary string into a structured range.

    Handles currency symbols/codes, `k`-suffixed thousands, and hourly/
    monthly/yearly period keywords. Falls back to a magnitude heuristic
    (values under 1000 read as hourly, otherwise yearly) when no period
    keyword is present, since ATS postings rarely label the period explicitly
    for small numbers like "$45 - $60".
    """
    if not raw or not raw.strip():
        return SalaryInfo(
            salary_min=None, salary_max=None, currency=None, period=SalaryPeriod.UNKNOWN
        )

    raw_lower = raw.lower()
    currency = _detect_currency(raw)
    period = _detect_period(raw_lower)
    numbers = _parse_numbers(raw)

    if not numbers:
        return SalaryInfo(salary_min=None, salary_max=None, currency=currency, period=period)

    salary_min = min(numbers)
    salary_max = max(numbers)

    if period == SalaryPeriod.UNKNOWN:
        period = SalaryPeriod.HOURLY if salary_max < 1000 else SalaryPeriod.YEARLY

    return SalaryInfo(
        salary_min=salary_min, salary_max=salary_max, currency=currency, period=period
    )


def parse_employment_type(raw: str | None) -> EmploymentType:
    """Best-effort keyword match of a free-text employment type string."""
    if not raw:
        return EmploymentType.UNKNOWN
    raw_lower = raw.lower()
    for employment_type, keywords in _EMPLOYMENT_TYPE_KEYWORDS.items():
        if any(keyword in raw_lower for keyword in keywords):
            return employment_type
    return EmploymentType.UNKNOWN


def parse_work_mode(
    *, location_raw: str | None, remote_flag: bool | None, description: str = ""
) -> WorkMode:
    """Best-effort work-mode detection from an explicit flag or keywords."""
    if remote_flag:
        return WorkMode.REMOTE

    haystack = f"{location_raw or ''} {description}".lower()
    if any(keyword in haystack for keyword in _REMOTE_KEYWORDS):
        return WorkMode.REMOTE
    if any(keyword in haystack for keyword in _HYBRID_KEYWORDS):
        return WorkMode.HYBRID
    if any(keyword in haystack for keyword in _ONSITE_KEYWORDS):
        return WorkMode.ONSITE
    return WorkMode.UNKNOWN


def parse_location(raw: str | None) -> LocationParts:
    """Best-effort split of a free-text location string into parts.

    Handles the common ATS shapes "City", "City, Region", and
    "City, Region, Country", plus a "Remote" special case. Ambiguous between
    US state and country in the two-part form (e.g. "Austin, TX" vs.
    "London, UK") — both are stored in `region` and disambiguation is left to
    a later phase.
    """
    if not raw or not raw.strip():
        return LocationParts(
            city=None, region=None, country=None, country_code=None, is_remote=False
        )

    stripped = raw.strip()
    if stripped.lower() in _REMOTE_KEYWORDS:
        return LocationParts(
            city=None, region=None, country=None, country_code=None, is_remote=True
        )

    parts = [p.strip() for p in stripped.split(",") if p.strip()]
    city = parts[0] if len(parts) >= 1 else None
    region = parts[1] if len(parts) >= 2 else None
    country = parts[2] if len(parts) >= 3 else None

    country_code = None
    for candidate in (country, region):
        if candidate and candidate.lower() in _COUNTRY_CODE_MAP:
            country_code = _COUNTRY_CODE_MAP[candidate.lower()]
            break

    return LocationParts(
        city=city, region=region, country=country, country_code=country_code, is_remote=False
    )


_TITLE_ID_SUFFIX_RE = re.compile(
    r"[\s\-–—|(]*(?:req|job|ref|id|posting)?\s*(?:id|#)?\s*[:#]?\s*[A-Z]{0,4}-?\d{3,}\)?\s*$",
    re.IGNORECASE,
)
_TITLE_WORK_MODE_PARENS_RE = re.compile(
    r"[\s\-–—|(]*\((?:remote|hybrid|on-?site|in[\s-]office)\)\s*$", re.IGNORECASE
)
_TITLE_TRAILING_SEP_RE = re.compile(r"[\s\-–—|]+$")
_WHITESPACE_RE = re.compile(r"\s+")

MIN_TITLE_LENGTH = 3
MIN_DESCRIPTION_WORDS = 30


def clean_title(raw_title: str) -> str:
    """Strip ATS noise from a job title so what's stored is presentable.

    Removes trailing requisition/job IDs (`"... - ID83932"`, `"(Req #1234)"`)
    and trailing work-mode parentheticals (`"(Remote)"`, which is redundant
    with the structured `work_mode` field), then collapses whitespace.
    Applied repeatedly since some ATSs stack both suffixes.
    """
    title = raw_title.strip()
    previous = None
    while title != previous:
        previous = title
        title = _TITLE_ID_SUFFIX_RE.sub("", title)
        title = _TITLE_WORK_MODE_PARENS_RE.sub("", title)
        title = _TITLE_TRAILING_SEP_RE.sub("", title)
    return _WHITESPACE_RE.sub(" ", title).strip()


def validate_posting(
    *,
    title: str,
    description: str,
    url: str,
    min_description_words: int = MIN_DESCRIPTION_WORDS,
) -> ValidationResult:
    """Reject postings too malformed to be worth storing.

    Sits between the adapter and the database: a bad title, an empty/near-
    empty description, or a missing URL means the adapter (or the upstream
    ATS) served something unusable, and it's better to skip that one posting
    with a clear reason than to silently store garbage or let normalization
    fail downstream with a less legible error.
    """
    errors: list[str] = []
    if not title or len(title.strip()) < MIN_TITLE_LENGTH:
        errors.append("title is missing or too short")
    if not url or not url.strip():
        errors.append("url is missing")
    word_count = len(description.split()) if description else 0
    if word_count < min_description_words:
        errors.append(
            f"description has only {word_count} words (minimum {min_description_words})"
        )
    return ValidationResult(is_valid=not errors, errors=errors)


def compute_content_hash(
    *, title: str, description: str, location_raw: str | None, salary_raw: str | None
) -> str:
    """Stable hash of the fields that matter for change detection.

    Used to skip re-normalization/re-embedding when an ATS re-serves an
    already-ingested job unchanged.
    """
    normalized = "|".join(
        [
            (title or "").strip().lower(),
            (description or "").strip().lower(),
            (location_raw or "").strip().lower(),
            (salary_raw or "").strip().lower(),
        ]
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


__all__ = [
    "SalaryInfo",
    "LocationParts",
    "ValidationResult",
    "parse_salary",
    "parse_employment_type",
    "parse_work_mode",
    "parse_location",
    "clean_title",
    "validate_posting",
    "compute_content_hash",
]
