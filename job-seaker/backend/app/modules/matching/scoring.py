"""Pure scoring/boost/narrative logic for the Matching module.

Deliberately free of DB/HTTP/I-O so it's unit-testable in isolation, mirroring
`app.modules.search.scoring`. Callers (the repository/service layer) are
responsible for extracting plain values from ORM objects before calling in.

Rule-boost functions return `float | None`:
    None  — the user hasn't set a preference for this dimension at all, so
            it's excluded from both the score (treated as 0) and the
            strengths/weaknesses narrative (there's nothing to say about a
            preference that doesn't exist).
    0.0   — a preference *is* set, but this job doesn't satisfy it.
    >0.0  — the preference is satisfied (fully, at the configured cap, or
            partially for the "adjacent experience bucket" case).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from app.modules.search.schemas import ExperienceLevel
from app.modules.search.scoring import classify_experience_level

# --- skill match ---------------------------------------------------------------


@dataclass(frozen=True)
class SkillMatchResult:
    matched: list[str]
    missing: list[str]
    score: float  # 0..100


def skill_match(
    resume_skills: Sequence[str], job_skills: Sequence[tuple[str, str]]
) -> SkillMatchResult:
    """Compare resume skills (freeform strings) against a job's normalized
    skill set (`(name, normalized_name)` pairs, from `Skill` rows).

    A job that lists no skills scores 100 on this dimension rather than 0 —
    a sparse extraction shouldn't be held against the candidate.
    """
    resume_normalized = {s.strip().lower() for s in resume_skills if s.strip()}
    if not job_skills:
        return SkillMatchResult(matched=[], missing=[], score=100.0)

    matched = [name for name, normalized in job_skills if normalized in resume_normalized]
    missing = [name for name, normalized in job_skills if normalized not in resume_normalized]
    score = (len(matched) / len(job_skills)) * 100.0
    return SkillMatchResult(matched=matched, missing=missing, score=score)


# --- rule boosts -----------------------------------------------------------------


def country_boost(
    preferred_countries: Sequence[str], job_country: str | None, *, cap: float
) -> float | None:
    if not preferred_countries:
        return None
    if not job_country:
        return 0.0
    normalized = {c.strip().lower() for c in preferred_countries if c.strip()}
    return cap if job_country.strip().lower() in normalized else 0.0


def work_mode_boost(
    preferred_work_modes: Sequence[str], job_work_mode: str, *, cap: float
) -> float | None:
    if not preferred_work_modes:
        return None
    normalized = {m.strip().lower() for m in preferred_work_modes if m.strip()}
    return cap if job_work_mode.strip().lower() in normalized else 0.0


def employment_type_boost(
    preferred_employment_types: Sequence[str], job_employment_type: str, *, cap: float
) -> float | None:
    if not preferred_employment_types:
        return None
    normalized = {t.strip().lower() for t in preferred_employment_types if t.strip()}
    return cap if job_employment_type.strip().lower() in normalized else 0.0


def salary_boost(
    *,
    expected_min: float | None,
    expected_max: float | None,
    expected_currency: str | None,
    job_min: float | None,
    job_max: float | None,
    job_currency: str | None,
    cap: float,
) -> float | None:
    if expected_min is None and expected_max is None:
        return None
    if job_min is None or job_max is None:
        return 0.0
    if expected_currency and job_currency and expected_currency != job_currency:
        return 0.0
    lo = expected_min if expected_min is not None else float("-inf")
    hi = expected_max if expected_max is not None else float("inf")
    overlaps = job_max >= lo and job_min <= hi
    return cap if overlaps else 0.0


_EXPERIENCE_LEVEL_ORDER = [
    ExperienceLevel.INTERNSHIP,
    ExperienceLevel.ENTRY,
    ExperienceLevel.MID,
    ExperienceLevel.SENIOR,
    ExperienceLevel.LEAD,
    ExperienceLevel.EXECUTIVE,
]


def experience_level_from_years(years_of_experience: int) -> ExperienceLevel:
    """Best-effort bucket for a candidate's total years of experience.

    Never returns INTERNSHIP — that bucket only makes sense as a job-posting
    classification (derived from title), not a candidate's years-of-experience
    tally.
    """
    if years_of_experience <= 2:
        return ExperienceLevel.ENTRY
    if years_of_experience <= 5:
        return ExperienceLevel.MID
    if years_of_experience <= 9:
        return ExperienceLevel.SENIOR
    if years_of_experience <= 14:
        return ExperienceLevel.LEAD
    return ExperienceLevel.EXECUTIVE


def experience_boost(
    years_of_experience: int | None, job_title: str, *, cap: float
) -> float | None:
    if years_of_experience is None:
        return None
    job_level = classify_experience_level(job_title)
    if job_level is None:
        return 0.0
    user_level = experience_level_from_years(years_of_experience)
    if user_level == job_level:
        return cap
    distance = abs(
        _EXPERIENCE_LEVEL_ORDER.index(user_level) - _EXPERIENCE_LEVEL_ORDER.index(job_level)
    )
    if distance == 1:
        return cap / 2
    return 0.0


# --- score composition -------------------------------------------------------------


def compute_match_score(
    *,
    semantic_score: float,
    skill_score: float,
    boosts: Iterable[float | None],
    semantic_weight: float,
    skill_weight: float,
) -> float:
    """Combine semantic similarity + skill overlap into a 0..100 base score,
    add rule-based boosts on top, and clamp to `[0, 100]`.
    """
    base = semantic_score * 100.0 * semantic_weight + skill_score * skill_weight
    total = base + sum(b for b in boosts if b is not None)
    return max(0.0, min(100.0, total))


# --- narrative strengths / weaknesses ----------------------------------------------

_BOOST_LABELS: dict[str, tuple[str, str]] = {
    "country": (
        "Located in one of your preferred countries.",
        "Job location is outside your preferred countries.",
    ),
    "work_mode": (
        "Work mode matches your preference.",
        "Work mode doesn't match your preference.",
    ),
    "employment_type": (
        "Employment type matches your preference.",
        "Employment type doesn't match your preference.",
    ),
    "salary": (
        "Salary range fits your expectations.",
        "Salary range doesn't fit your expectations.",
    ),
    "experience": (
        "Experience level aligns with this role.",
        "Experience level doesn't align well with this role.",
    ),
}


@dataclass(frozen=True)
class StrengthsWeaknesses:
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)


def build_strengths_and_weaknesses(
    *,
    semantic_score: float,
    skill_result: SkillMatchResult,
    boosts: dict[str, float | None],
    top_n_missing: int = 5,
) -> StrengthsWeaknesses:
    strengths: list[str] = []
    weaknesses: list[str] = []

    if semantic_score >= 0.8:
        strengths.append("Your resume content strongly aligns with this role's responsibilities.")
    elif semantic_score < 0.4:
        weaknesses.append("Your resume doesn't closely match this role's description.")

    job_skill_count = len(skill_result.matched) + len(skill_result.missing)
    if job_skill_count > 0:
        if skill_result.score >= 70:
            detail = (
                f", including {', '.join(skill_result.matched[:3])}" if skill_result.matched else ""
            )
            strengths.append(
                f"You match {len(skill_result.matched)}/{job_skill_count} required skills{detail}."
            )
        elif skill_result.score < 40:
            weaknesses.append("Your resume shows limited overlap with this role's required skills.")
    if skill_result.missing:
        weaknesses.append(
            f"Missing skills this role lists: {', '.join(skill_result.missing[:top_n_missing])}."
        )

    for key, value in boosts.items():
        if value is None or key not in _BOOST_LABELS:
            continue
        strength_text, weakness_text = _BOOST_LABELS[key]
        if value > 0:
            strengths.append(strength_text)
        else:
            weaknesses.append(weakness_text)

    return StrengthsWeaknesses(strengths=strengths, weaknesses=weaknesses)


__all__ = [
    "SkillMatchResult",
    "skill_match",
    "country_boost",
    "work_mode_boost",
    "employment_type_boost",
    "salary_boost",
    "experience_level_from_years",
    "experience_boost",
    "compute_match_score",
    "StrengthsWeaknesses",
    "build_strengths_and_weaknesses",
]
