"""Search module — heuristic v1 "AI signals" (Hiring Probability, Application
Difficulty, Expected ATS Score).

No model in the schema captures these signals today (unlike `match_score`,
which is a real computation over `matching.ResumeJobMatch`), so this module
provides a deterministic, formula-based v1 — same "pure function, no I/O"
convention as `search/scoring.py`. Every function here takes already-fetched
primitives (never a live DB session, never an ORM relationship it has to
lazy-load) so it stays trivially unit-testable and is designed to be a
drop-in-replaceable seam: swap the body of any one function for a real ML
model call later without touching any caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DifficultyLabel = Literal["easy", "moderate", "hard"]

# Difficulty score bucket boundaries (see `compute_application_difficulty`).
_DIFFICULTY_EASY_MAX = 35.0
_DIFFICULTY_MODERATE_MAX = 65.0

# Hiring-probability clamp range — kept away from the 0/100 extremes since a
# heuristic estimate should never claim total certainty either way.
_HIRING_PROB_FLOOR = 5.0
_HIRING_PROB_CEILING = 95.0

# Recency decays linearly from this score at day 0...
_RECENCY_SCORE_AT_DAY_ZERO = 100.0
# ...down to this score once `SEARCH_HIRING_PROB_RECENCY_DECAY_DAYS` (or more)
# have elapsed since posting.
_RECENCY_SCORE_AT_FLOOR = 20.0

# Cadence score used when a company's `hiring_frequency_per_month` is unknown
# (no `CompanyIntelligence` row, or the field itself is `None`) — neutral
# rather than penalizing/rewarding a company we simply have no signal for.
_CADENCE_NEUTRAL_SCORE = 50.0
# `hiring_frequency_per_month` value that maps to a full 100 cadence score.
# Chosen as a round "hires roughly every week" reference point; anything at
# or above it saturates rather than continuing to climb unbounded.
_CADENCE_SATURATION_FREQUENCY = 4.0

_RECENCY_WEIGHT = 0.6
_CADENCE_WEIGHT = 0.4

# Seniority keywords that push a posting toward "requirements-heavy" in the
# application-difficulty score — reuses the same lexical-signal approach as
# `search/scoring.py::EXPERIENCE_LEVEL_TITLE_PATTERNS`, kept separate here
# since difficulty cares about *any* seniority marker, not which level.
_SENIOR_TITLE_KEYWORDS = (
    "senior",
    "sr.",
    "sr ",
    "staff",
    "principal",
    "lead ",
    "director",
    "head of",
    "vp ",
    "vice president",
    "chief",
)

# Description-length thresholds (characters) contributing to difficulty —
# a very long posting usually spells out more requirements/process detail.
_DESCRIPTION_LENGTH_MODERATE_CHARS = 1500
_DESCRIPTION_LENGTH_LONG_CHARS = 3000


def _clamp(value: float, *, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_hiring_probability(
    *,
    hiring_frequency_per_month: float | None,
    days_since_posted: float,
    recency_decay_days: int,
) -> float:
    """Heuristic v1 "Hiring Probability" (0-100, higher = more likely the
    posting is actively being filled soon).

    `0.6 * recency_score + 0.4 * cadence_score`:
      - `recency_score` decays *linearly* from 100 (posted today) to 20 (at
        or beyond `recency_decay_days`) — a fresh posting is more likely to
        still be actively screening than a stale one.
      - `cadence_score` is derived from the posting company's historical
        `CompanyIntelligence.hiring_frequency_per_month` (hires/month),
        saturating at `_CADENCE_SATURATION_FREQUENCY`; `None` (no company
        intelligence on file) maps to a neutral 50, not a penalty.

    Clamped to `[5, 95]` — a pure heuristic should never claim 0% or 100%
    certainty. Swappable: replace this body with a real model's predict()
    call; the 0-100 contract and the caller side (`heuristics.*` import in
    `job_feed_service.py`) don't need to change.
    """
    days_since_posted = max(0.0, days_since_posted)
    decay_days = max(1, recency_decay_days)
    recency_fraction = min(1.0, days_since_posted / decay_days)
    recency_score = _RECENCY_SCORE_AT_DAY_ZERO - recency_fraction * (
        _RECENCY_SCORE_AT_DAY_ZERO - _RECENCY_SCORE_AT_FLOOR
    )

    if hiring_frequency_per_month is None:
        cadence_score = _CADENCE_NEUTRAL_SCORE
    else:
        frequency = max(0.0, hiring_frequency_per_month)
        cadence_score = min(100.0, (frequency / _CADENCE_SATURATION_FREQUENCY) * 100.0)

    probability = _RECENCY_WEIGHT * recency_score + _CADENCE_WEIGHT * cadence_score
    return round(_clamp(probability, low=_HIRING_PROB_FLOOR, high=_HIRING_PROB_CEILING), 2)


def compute_application_difficulty(
    *,
    required_skill_count: int,
    experience_level_title: str,
    description_length: int,
    skill_count_threshold: int,
) -> tuple[DifficultyLabel, float]:
    """Heuristic v1 "Application Difficulty" — a 0-100 score plus its
    `easy` (<35) / `moderate` (35-65) / `hard` (>65) bucket label.

    Three independent, additive signals (each contributes up to ~1/3 of the
    100-point scale, so no single signal alone can push a posting past
    `moderate`):
      - **Skill-tag count**: more than `skill_count_threshold` required
        skills is treated as "requirements-heavy" and scores higher,
        scaling linearly up to double the threshold (saturating at 35pts).
      - **Seniority keywords in the title**: a senior/staff/lead/director+
        title (see `_SENIOR_TITLE_KEYWORDS`) adds a flat 30pts — a posting
        explicitly seeking seniority usually implies a higher expectations
        bar during the application/interview process itself.
      - **Description length**: a long posting (more process/requirement
        detail spelled out) contributes up to 35pts, scaling between
        `_DESCRIPTION_LENGTH_MODERATE_CHARS` and `_DESCRIPTION_LENGTH_LONG_CHARS`.

    Swappable: this is a lexical/count-based proxy for "how hard is this
    application," not a model trained on real outcome data — replace the
    body with a real model call later; the `(label, score)` contract stays.
    """
    skill_component = 0.0
    if required_skill_count > skill_count_threshold:
        overflow = required_skill_count - skill_count_threshold
        skill_component = min(35.0, (overflow / max(1, skill_count_threshold)) * 35.0)

    title_lower = experience_level_title.lower()
    seniority_component = 30.0 if any(kw in title_lower for kw in _SENIOR_TITLE_KEYWORDS) else 0.0

    if description_length <= _DESCRIPTION_LENGTH_MODERATE_CHARS:
        length_component = 0.0
    elif description_length >= _DESCRIPTION_LENGTH_LONG_CHARS:
        length_component = 35.0
    else:
        span = _DESCRIPTION_LENGTH_LONG_CHARS - _DESCRIPTION_LENGTH_MODERATE_CHARS
        length_component = (
            (description_length - _DESCRIPTION_LENGTH_MODERATE_CHARS) / span
        ) * 35.0

    raw_score = skill_component + seniority_component + length_component
    score = round(_clamp(raw_score, low=0.0, high=100.0), 2)

    if score < _DIFFICULTY_EASY_MAX:
        label: DifficultyLabel = "easy"
    elif score <= _DIFFICULTY_MODERATE_MAX:
        label = "moderate"
    else:
        label = "hard"
    return label, score


def estimate_ats_score(
    *,
    resume_ats_baseline_score: float | None,
    missing_skill_count: int,
    total_skill_count: int,
) -> float | None:
    """Heuristic v1 "Expected ATS Score" for a specific job, projected from
    the user's resume baseline ATS score.

    `baseline * (1 - 0.5 * missing_ratio)`, clamped `[0, 100]` — a job
    requiring skills absent from the resume is projected to score lower
    than the resume's own baseline (from `ResumeVersion.ats_baseline_score`,
    a job-agnostic score), proportional to how much of that job's skill list
    is missing. Returns `None` (not zero) when the user has no resume yet —
    "no data" and "projected zero" are different states the caller/UI must
    distinguish.

    Swappable: this is a linear proxy, not the real
    `ats_resume_generator` keyword-match scoring engine run against this
    specific job description — replace the body with a call into that (or a
    trained model) later without changing the `float | None` contract.
    """
    if resume_ats_baseline_score is None:
        return None
    if total_skill_count <= 0:
        missing_ratio = 0.0
    else:
        missing_ratio = min(1.0, max(0.0, missing_skill_count) / total_skill_count)
    projected = resume_ats_baseline_score * (1 - 0.5 * missing_ratio)
    return round(_clamp(projected, low=0.0, high=100.0), 2)


@dataclass(frozen=True)
class JobAISignalValues:
    """Plain container bundling all three heuristic outputs for one job —
    convenience for callers building `search.schemas.JobAISignals`.
    """

    hiring_probability: float
    application_difficulty: DifficultyLabel
    application_difficulty_score: float
    expected_ats_score: float | None


__all__ = [
    "DifficultyLabel",
    "JobAISignalValues",
    "compute_hiring_probability",
    "compute_application_difficulty",
    "estimate_ats_score",
]
