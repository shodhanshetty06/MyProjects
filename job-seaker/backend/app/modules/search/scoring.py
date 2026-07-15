"""Pure ranking/fusion/heuristic logic for the Search module.

Deliberately free of DB/HTTP/I-O so it's unit-testable in isolation. Hybrid
fusion uses a normalized weighted sum (not reciprocal-rank-fusion): the
`match_score` sort mode needs the actual cosine-similarity magnitude, not
just a rank, so one magnitude-based scheme serves both `relevance` and
`match_score` sorting instead of maintaining two scoring paths.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.modules.search.schemas import DatePosted, ExperienceLevel


@dataclass(frozen=True)
class ScoreBreakdown:
    job_id: uuid.UUID
    keyword_score: float | None
    semantic_score: float | None
    hybrid_score: float


def normalize_keyword_scores(raw: dict[uuid.UUID, float]) -> dict[uuid.UUID, float]:
    """Min-max normalize raw `ts_rank_cd` scores to the 0..1 range."""
    if not raw:
        return {}
    lo, hi = min(raw.values()), max(raw.values())
    if hi <= lo:
        return {job_id: (1.0 if value > 0 else 0.0) for job_id, value in raw.items()}
    return {job_id: (value - lo) / (hi - lo) for job_id, value in raw.items()}


def fuse_scores(
    keyword: dict[uuid.UUID, float],
    semantic: dict[uuid.UUID, float],
    *,
    w_keyword: float,
    w_semantic: float,
) -> dict[uuid.UUID, ScoreBreakdown]:
    """Fuse normalized keyword rank + semantic similarity into one score.

    A job present in only one candidate set is scored using that set's
    contribution alone (missing side treated as 0), not excluded.
    """
    keyword_norm = normalize_keyword_scores(keyword)
    job_ids = set(keyword_norm) | set(semantic)
    result: dict[uuid.UUID, ScoreBreakdown] = {}
    for job_id in job_ids:
        keyword_score = keyword_norm.get(job_id)
        semantic_score = semantic.get(job_id)
        hybrid_score = w_keyword * (keyword_score or 0.0) + w_semantic * (semantic_score or 0.0)
        result[job_id] = ScoreBreakdown(job_id, keyword_score, semantic_score, hybrid_score)
    return result


# --- Experience-level heuristic (no `Job.experience_level` column exists;
# derived from the job title as a pragmatic stand-in — see search/README
# note in repository.py for the long-term fix). ---

EXPERIENCE_LEVEL_TITLE_PATTERNS: dict[ExperienceLevel, list[str]] = {
    ExperienceLevel.INTERNSHIP: ["intern", "internship"],
    ExperienceLevel.ENTRY: ["junior", "entry level", "entry-level", "graduate", "new grad"],
    ExperienceLevel.SENIOR: ["senior", "sr.", "sr "],
    ExperienceLevel.LEAD: ["lead ", "staff", "principal"],
    ExperienceLevel.EXECUTIVE: ["director", "vp ", "vice president", "head of", "chief"],
}

# Levels checked in this order so more specific/senior keywords win over a
# job title that could ambiguously match more than one bucket.
_EXPERIENCE_LEVEL_PRIORITY = [
    ExperienceLevel.EXECUTIVE,
    ExperienceLevel.LEAD,
    ExperienceLevel.SENIOR,
    ExperienceLevel.INTERNSHIP,
    ExperienceLevel.ENTRY,
]


def matches_experience_level(title: str, level: ExperienceLevel) -> bool:
    if level == ExperienceLevel.MID:
        return not any(
            matches_experience_level(title, other) for other in _EXPERIENCE_LEVEL_PRIORITY
        )
    patterns = EXPERIENCE_LEVEL_TITLE_PATTERNS.get(level, [])
    title_lower = title.lower()
    return any(pattern in title_lower for pattern in patterns)


def classify_experience_level(title: str) -> ExperienceLevel | None:
    """Best-effort experience level for display, derived from the title.

    Returns `None` (unknown) when no signal is present, rather than
    defaulting to MID — MID is reserved for filtering ("not junior/senior/
    lead/executive/intern"), not for display of an undetermined level.
    """
    for level in _EXPERIENCE_LEVEL_PRIORITY:
        if matches_experience_level(title, level):
            return level
    return None


def date_posted_cutoff(date_posted: DatePosted, *, now: datetime | None = None) -> datetime | None:
    """Resolve a `DatePosted` bucket into an absolute `posted_at >=` cutoff."""
    if date_posted == DatePosted.ANY:
        return None
    reference = now or datetime.now(UTC)
    deltas = {
        DatePosted.LAST_24H: timedelta(hours=24),
        DatePosted.LAST_7D: timedelta(days=7),
        DatePosted.LAST_30D: timedelta(days=30),
    }
    return reference - deltas[date_posted]
