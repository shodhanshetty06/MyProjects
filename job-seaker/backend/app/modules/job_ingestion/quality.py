"""Job Quality Score — pure, deterministic scoring (Feature 5).

Deliberately rule-based rather than a trained/learned model: the score has
to be explainable ("why is this 82/100?") and stable as the job corpus
grows, and a model fit to historical postings risks overfitting to whatever
quirks happen to be in that data (e.g. learning "Series B startups score
low" from a small sample) rather than measuring actual posting quality.
Weights are config-driven (`Settings.JOB_QUALITY_WEIGHT_*`) so they can be
tuned without a code change, but the scoring logic itself never trains on
data.

No DB session, no I/O — mirrors `normalization.py`. `tasks.py` is the only
caller, supplying already-loaded `Job`/`Company` ORM instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from app.core.config import Settings
from app.modules.job_ingestion.models import ATSPlatform, Company, Job, JobStatus, WorkMode
from app.modules.job_ingestion.normalization import MIN_DESCRIPTION_WORDS


@dataclass(frozen=True)
class QualityReason:
    label: str
    positive: bool
    weight: float

    def to_dict(self) -> dict:
        return {"label": self.label, "positive": self.positive, "weight": self.weight}


@dataclass(frozen=True)
class QualityScoreResult:
    score: int
    reasons: list[QualityReason]

    def reasons_as_dicts(self) -> list[dict]:
        return [reason.to_dict() for reason in self.reasons]


def _is_company_verified(company: Company) -> bool:
    """A company is "verified" once it has either a real website or a
    recognized ATS integration configured — both are signals a human
    actually set this company up, rather than it being a bare stub."""
    if company.website:
        return True
    intelligence = company.intelligence
    return intelligence is not None and intelligence.ats_platform != ATSPlatform.UNKNOWN


def compute_quality_score(
    job: Job, company: Company, *, settings: Settings, now: datetime | None = None
) -> QualityScoreResult:
    now = now or datetime.now(UTC)
    reasons: list[QualityReason] = []

    def _check(label: str, positive: bool, weight: float) -> None:
        reasons.append(QualityReason(label=label, positive=positive, weight=weight))

    has_salary = job.salary_min is not None or job.salary_max is not None
    _check("Salary listed", has_salary, settings.JOB_QUALITY_WEIGHT_SALARY)

    word_count = len(job.description.split()) if job.description else 0
    has_good_description = word_count >= MIN_DESCRIPTION_WORDS
    _check(
        "Detailed description" if has_good_description else "Poor description",
        has_good_description,
        settings.JOB_QUALITY_WEIGHT_DESCRIPTION,
    )

    verified = _is_company_verified(company)
    _check("Verified company", verified, settings.JOB_QUALITY_WEIGHT_VERIFIED_COMPANY)

    is_active = job.status == JobStatus.ACTIVE
    _check("Active", is_active, settings.JOB_QUALITY_WEIGHT_ACTIVE)

    has_skills = bool(job.required_skills or job.preferred_skills or job.skills)
    _check("Skills identified", has_skills, settings.JOB_QUALITY_WEIGHT_SKILLS)

    has_location_signal = job.work_mode != WorkMode.UNKNOWN or job.location_id is not None
    _check(
        "Remote" if job.work_mode == WorkMode.REMOTE else "Location specified",
        has_location_signal,
        settings.JOB_QUALITY_WEIGHT_LOCATION,
    )

    is_recent = False
    if job.posted_at is not None:
        reference = job.posted_at
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=UTC)
        age_days = (now - reference).days
        is_recent = age_days <= settings.JOB_QUALITY_RECENCY_DAYS
    _check("Recently posted", is_recent, settings.JOB_QUALITY_WEIGHT_RECENCY)

    earned = sum(reason.weight for reason in reasons if reason.positive)
    total = sum(reason.weight for reason in reasons) or 1.0
    score = round(100 * earned / total)
    return QualityScoreResult(score=score, reasons=reasons)


__all__ = ["QualityReason", "QualityScoreResult", "compute_quality_score"]
