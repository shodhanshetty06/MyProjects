"""Search module — Ranking Engine (multi-factor weighted composite score).

Same "pure function, no I/O" convention as `heuristics.py`/`scoring.py`:
every input here is an already-fetched primitive, every output is a plain
0-100 float. This is a deterministic, documented, config-weighted formula —
not a model trained on this project's own (small) dataset — so it stays
swappable for a real learned ranker later without any caller change, exactly
like `heuristics.py`'s existing signals.

The six factors and their default weights (`RankingWeights`, driven by
`Settings.RANKING_WEIGHT_*`) match the product spec:
    Resume Match        40%
    Hiring Probability   20%
    Salary               15%
    Recent Posting       10%
    Company Quality      10%
    Competition            5%
"""

from __future__ import annotations

from dataclasses import dataclass

# Neutral fallback used for company quality when neither a
# `CompanyIntelligence.company_rating` nor an average job `quality_score` is
# on file — "no signal," not a penalty.
_COMPANY_QUALITY_NEUTRAL = 50.0

# Company-rating scale (0.00-5.00, see `CompanyIntelligence.company_rating`)
# rescaled to 0-100.
_COMPANY_RATING_MAX = 5.0

# Competition saturates (fewer strong competitors = better opportunity, more
# = worse) at this many other strong-matching candidates; beyond it the
# competition score bottoms out rather than continuing to fall unbounded.
_COMPETITION_SATURATION_COUNT = 20

# Recency decays linearly from 100 (posted today) to this floor once
# `recency_decay_days` (or more) have elapsed — mirrors
# `heuristics.compute_hiring_probability`'s recency component, extracted
# here as its own standalone signal so the ranking engine and the hiring-
# probability heuristic can each use it independently.
_RECENCY_SCORE_AT_DAY_ZERO = 100.0
_RECENCY_SCORE_AT_FLOOR = 10.0


def _clamp(value: float, *, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class RankingWeights:
    resume_match: float
    hiring_probability: float
    salary: float
    recency: float
    company_quality: float
    competition: float

    @property
    def total(self) -> float:
        return (
            self.resume_match
            + self.hiring_probability
            + self.salary
            + self.recency
            + self.company_quality
            + self.competition
        )


@dataclass(frozen=True)
class RankingBreakdown:
    """Per-factor 0-100 inputs plus the final weighted score — returned
    alongside the score so the UI can show "why this is #1" (pairs
    naturally with Feature 1's match-explanation surface).
    """

    resume_match: float | None
    hiring_probability: float
    salary: float
    recency: float
    company_quality: float
    competition: float
    score: float


def compute_recency_score(*, days_since_posted: float, recency_decay_days: int) -> float:
    """0-100, linearly decaying from 100 (posted today) to 10 (posted
    `recency_decay_days` or more days ago). Standalone extraction of the
    recency component already used inside
    `heuristics.compute_hiring_probability` — kept here as its own signal so
    the ranking engine doesn't have to reverse-engineer it out of hiring
    probability.
    """
    days_since_posted = max(0.0, days_since_posted)
    decay_days = max(1, recency_decay_days)
    fraction = min(1.0, days_since_posted / decay_days)
    score = _RECENCY_SCORE_AT_DAY_ZERO - fraction * (
        _RECENCY_SCORE_AT_DAY_ZERO - _RECENCY_SCORE_AT_FLOOR
    )
    return round(_clamp(score), 2)


def compute_salary_percentile(
    salary: float | None, *, comparison_salaries: list[float]
) -> float:
    """0-100 percentile rank of `salary` within `comparison_salaries` (the
    current candidate/result set, not a global distribution — no separate
    salary-benchmark data source exists, and ranking jobs *relative to what
    the user is currently looking at* is the more honest claim to make).
    `None` salary (not disclosed) or fewer than 2 comparison points both
    return the neutral midpoint rather than a fabricated extreme.
    """
    if salary is None or len(comparison_salaries) < 2:
        return 50.0
    below_or_equal = sum(1 for s in comparison_salaries if s <= salary)
    percentile = (below_or_equal / len(comparison_salaries)) * 100.0
    return round(_clamp(percentile), 2)


def compute_company_quality(
    *, company_rating: float | None, avg_job_quality_score: float | None
) -> float:
    """0-100 "Company Quality" heuristic v1. Prefers a real aggregate rating
    (`CompanyIntelligence.company_rating`, 0-5 scale) when one has been
    populated; falls back to the company's average posting `quality_score`
    (already computed for every job by `job_ingestion/quality.py`, so this
    needs no new data source); falls back further to a neutral midpoint when
    neither signal exists yet (most companies today) — never a penalty for
    missing data.
    """
    if company_rating is not None:
        return round(_clamp((company_rating / _COMPANY_RATING_MAX) * 100.0), 2)
    if avg_job_quality_score is not None:
        return round(_clamp(avg_job_quality_score), 2)
    return _COMPANY_QUALITY_NEUTRAL


def compute_competition_score(*, strong_match_count: int) -> float:
    """0-100 "Competition" signal — higher score means *less* competition
    (a better opportunity), consistent with every other ranking factor here
    being "higher is better." `strong_match_count` is the number of other
    users whose resume also scores at/above the notify threshold for this
    job (see `search/repository.py::get_competition_counts`) — the only
    applicant-pressure proxy available in-schema (no scraped applicant
    counts exist anywhere).
    """
    count = max(0, strong_match_count)
    saturation = max(1, _COMPETITION_SATURATION_COUNT)
    fraction = min(1.0, count / saturation)
    score = 100.0 - fraction * 100.0
    return round(_clamp(score), 2)


def compute_ranking_score(
    *,
    resume_match: float | None,
    hiring_probability: float,
    salary: float,
    recency: float,
    company_quality: float,
    competition: float,
    weights: RankingWeights,
) -> RankingBreakdown:
    """Weighted composite of the six factors, each already 0-100.

    When `resume_match` is `None` (user has no resume/no computed match for
    this job yet), its weight is redistributed proportionally across the
    remaining five factors rather than treated as 0 — a job shouldn't be
    unfairly buried in ranking just because the user hasn't uploaded a
    resume yet; the remaining signals simply carry the full 100% instead.
    """
    factors = {
        "hiring_probability": (hiring_probability, weights.hiring_probability),
        "salary": (salary, weights.salary),
        "recency": (recency, weights.recency),
        "company_quality": (company_quality, weights.company_quality),
        "competition": (competition, weights.competition),
    }

    if resume_match is not None:
        factors["resume_match"] = (resume_match, weights.resume_match)

    weight_sum = sum(w for _, w in factors.values())
    if weight_sum <= 0:
        score = 0.0
    else:
        score = sum(value * (weight / weight_sum) for value, weight in factors.values())

    return RankingBreakdown(
        resume_match=resume_match,
        hiring_probability=hiring_probability,
        salary=salary,
        recency=recency,
        company_quality=company_quality,
        competition=competition,
        score=round(_clamp(score), 2),
    )


__all__ = [
    "RankingWeights",
    "RankingBreakdown",
    "compute_recency_score",
    "compute_salary_percentile",
    "compute_company_quality",
    "compute_competition_score",
    "compute_ranking_score",
]
