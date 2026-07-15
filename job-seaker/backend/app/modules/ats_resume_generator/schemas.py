"""ATS Resume Generator module — Pydantic request/response contracts."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.modules.ats_resume_generator.models import ATSFileFormat, ATSVersionStatus
from app.modules.resume_parser.schemas import (
    CertificationItem,
    EducationItem,
    ExperienceItem,
    LanguageItem,
    ProjectItem,
)


class OptimizedContent(BaseModel):
    """Claude's structured-output contract for the ATS optimize pass.

    Deliberately limited to summary/skills/projects/keywords — the only
    sections Claude is allowed to rewrite. `experience`, `education`,
    `certifications`, `languages`, and `achievements` are never sent to the
    model; they're copied verbatim from the source resume's parsed data so
    work history can never be fabricated. `model_json_schema()` on this
    class is the schema handed to `AIService.extract_structured`.
    """

    summary: str = Field(
        description=(
            "A concise 2-4 sentence professional summary, rewritten to align with the "
            "target job description using only facts already present in the source resume."
        )
    )
    skills: list[str] = Field(
        default_factory=list,
        description=(
            "Reprioritized/regrouped skill list. Every entry must already appear "
            "(verbatim or as a clear synonym/abbreviation) in the source resume text — "
            "never add a skill the candidate hasn't demonstrated."
        ),
    )
    projects: list[ProjectItem] = Field(
        default_factory=list,
        description=(
            "The same projects that appear in the source resume, with description/highlights "
            "rewritten for clarity and keyword alignment. Do not invent new projects, "
            "technologies, or outcomes, and do not drop a project that was present."
        ),
    )
    keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Important skills, technologies, and role-specific terms extracted from the "
            "job description — used to score keyword coverage, not written into the resume verbatim."
        ),
    )


class ATSResumeCreateRequest(BaseModel):
    resume_version_id: uuid.UUID
    job_description: str = Field(min_length=20)
    title: str | None = Field(default=None, max_length=255)
    job_title: str | None = Field(default=None, max_length=255)
    company_name: str | None = Field(default=None, max_length=255)
    target_job_id: uuid.UUID | None = None


class ATSResumeVersionCreateRequest(BaseModel):
    """Create a new version — re-optimize, optionally against a new job
    description and/or a different source resume version. Unset fields fall
    back to whatever the `ATSResume`'s current version used.
    """

    job_description: str | None = Field(default=None, min_length=20)
    resume_version_id: uuid.UUID | None = None


class ATSResumeUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class ATSResumeVersionSummary(BaseModel):
    id: uuid.UUID
    version_number: int
    is_current: bool
    status: ATSVersionStatus
    keyword_score: float | None = None
    ats_score: float | None = None
    error_message: str | None = None
    available_formats: list[ATSFileFormat] = Field(default_factory=list)
    created_at: datetime


class ATSResumeVersionDetail(ATSResumeVersionSummary):
    job_description: str
    source_resume_version_id: uuid.UUID
    original_summary: str
    optimized_summary: str
    original_skills: list[str]
    optimized_skills: list[str]
    original_projects: list[ProjectItem]
    optimized_projects: list[ProjectItem]
    experience: list[ExperienceItem]
    education: list[EducationItem]
    certifications: list[CertificationItem]
    languages: list[LanguageItem]
    achievements: list[str]
    keywords: list[str]
    matched_keywords: list[str]
    missing_keywords: list[str]
    score_breakdown: dict
    model_used: str | None = None
    updated_at: datetime


class ATSResumeRead(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    source_resume_id: uuid.UUID
    target_job_id: uuid.UUID | None
    title: str
    job_title: str | None
    company_name: str | None
    created_at: datetime
    updated_at: datetime
    current_version: ATSResumeVersionSummary | None = None


class TextDiffSegmentSchema(BaseModel):
    op: str
    text: str


class ListDiffSchema(BaseModel):
    added: list[str]
    removed: list[str]
    unchanged: list[str]


class ProjectDiffSchema(BaseModel):
    name: str
    status: str
    description_diff: list[TextDiffSegmentSchema]
    highlights_diff: ListDiffSchema


class ATSResumeVersionDiff(BaseModel):
    compare_to: str
    summary_diff: list[TextDiffSegmentSchema]
    skills_diff: ListDiffSchema
    projects_diff: list[ProjectDiffSchema]


class ATSResumeDownloadResponse(BaseModel):
    format: ATSFileFormat
    download_url: str


__all__ = [
    "OptimizedContent",
    "ATSResumeCreateRequest",
    "ATSResumeVersionCreateRequest",
    "ATSResumeUpdateRequest",
    "ATSResumeVersionSummary",
    "ATSResumeVersionDetail",
    "ATSResumeRead",
    "TextDiffSegmentSchema",
    "ListDiffSchema",
    "ProjectDiffSchema",
    "ATSResumeVersionDiff",
    "ATSResumeDownloadResponse",
]
