"""Resume Parser module — Pydantic request/response contracts."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.modules.resume_parser.models import ResumeContentType, ResumeVersionStatus


class ExperienceItem(BaseModel):
    company: str = Field(description="Employer name")
    title: str = Field(description="Job title")
    location: str | None = Field(default=None, description="City/region of the role")
    start_date: str | None = Field(default=None, description="Start date, as written on the resume")
    end_date: str | None = Field(default=None, description="End date, or null/'Present' if ongoing")
    is_current: bool = Field(default=False, description="Whether this is the person's current role")
    description: str | None = Field(default=None, description="Role summary")
    highlights: list[str] = Field(default_factory=list, description="Bullet-point achievements")


class EducationItem(BaseModel):
    institution: str
    degree: str | None = None
    field_of_study: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    gpa: str | None = None


class ProjectItem(BaseModel):
    name: str
    description: str | None = None
    technologies: list[str] = Field(default_factory=list)
    url: str | None = None


class CertificationItem(BaseModel):
    name: str
    issuer: str | None = None
    issue_date: str | None = None
    expiry_date: str | None = None
    credential_id: str | None = None


class LanguageItem(BaseModel):
    language: str
    proficiency: str | None = Field(default=None, description="e.g. native, fluent, conversational")


class ParsedResumeData(BaseModel):
    """Structured extraction contract.

    `model_json_schema()` on this class is the schema handed to
    `AIService.extract_structured` — it is both the API response shape and
    the Claude structured-output contract, so the two can never drift apart.
    """

    summary: str = Field(description="A concise 2-4 sentence professional summary")
    skills: list[str] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)
    education: list[EducationItem] = Field(default_factory=list)
    projects: list[ProjectItem] = Field(default_factory=list)
    certifications: list[CertificationItem] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    languages: list[LanguageItem] = Field(default_factory=list)


class ResumeVersionSummary(BaseModel):
    """Lightweight version info embedded in `ResumeRead`."""

    id: uuid.UUID
    version_number: int
    is_current: bool
    status: ResumeVersionStatus
    content_type: ResumeContentType
    original_filename: str
    file_size_bytes: int
    error_message: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ResumeVersionDetail(ResumeVersionSummary):
    """Version info plus parsed data, once parsing has completed."""

    parsed_data: ParsedResumeData | None = None
    has_embedding: bool = False


class ResumeRead(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    current_version: ResumeVersionSummary | None = None

    model_config = {"from_attributes": True}


class ResumeUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)


__all__ = [
    "ExperienceItem",
    "EducationItem",
    "ProjectItem",
    "CertificationItem",
    "LanguageItem",
    "ParsedResumeData",
    "ResumeVersionSummary",
    "ResumeVersionDetail",
    "ResumeRead",
    "ResumeUpdateRequest",
]
