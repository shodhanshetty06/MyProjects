"""Resume Parser module — Pydantic request/response contracts."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.modules.resume_parser.models import ResumeContentType, ResumeVersionStatus


class ContactInfo(BaseModel):
    # Required-with-no-default in the extraction *prompt* sense (the system
    # prompt instructs the model to never leave it blank), but defaulted to
    # ""  here so reading back rows from before this field existed (empty
    # `contact` JSONB) or a version whose source resume genuinely had no
    # extractable name doesn't 500 the API — the pre-export validator
    # (`validation.py`) is what actually blocks a nameless resume from
    # being exported, not this schema.
    full_name: str = Field(
        default="",
        description=(
            "The candidate's full name, exactly as written — usually the largest/most "
            "prominent text at the top of the resume. Never leave this blank; if no name "
            "heading is present, use the name as it appears anywhere in the document (e.g. "
            "in an email signature)."
        ),
    )
    phone: str | None = Field(default=None, description="Phone number, exactly as written")
    email: str | None = Field(default=None, description="Email address, exactly as written")
    linkedin_url: str | None = Field(
        default=None, description="LinkedIn profile URL, if present in the header/contact block"
    )
    github_url: str | None = Field(
        default=None, description="GitHub profile URL, if present in the header/contact block"
    )
    portfolio_url: str | None = Field(
        default=None,
        description="Personal website/portfolio URL, if present (distinct from LinkedIn/GitHub)",
    )
    location: str | None = Field(
        default=None, description="City/region as written, e.g. 'Bengaluru, India'"
    )


class ExperienceItem(BaseModel):
    company: str = Field(description="Employer name")
    title: str = Field(
        description=(
            "Job title. If the entry (e.g. an internship) has no explicit job title, use the "
            "nearest role/department/project label instead (e.g. 'Department: Tool Management' "
            "-> 'Tool Management') rather than omitting the entry entirely."
        )
    )
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
    gpa: str | None = Field(
        default=None,
        description=(
            "The grade value only, exactly as written on the resume (e.g. '3.8', '8.5', "
            "'85%') — never the words 'GPA', 'CGPA', or 'Percentage'. Leave null if no "
            "grade is present on the resume."
        ),
    )
    gpa_type: Literal["gpa", "cgpa", "percentage"] | None = Field(
        default=None,
        description=(
            "How the grade in `gpa` is scaled: 'gpa' (out of 4), 'cgpa' (out of 10), or "
            "'percentage' (common for schooling up to PU/12th grade). Infer from context "
            "(e.g. a value like '8.5' next to an Indian university is 'cgpa'; '92%' is "
            "'percentage'). Leave null if unclear or no grade is present."
        ),
    )


class ProjectItem(BaseModel):
    name: str = Field(description="Project title/name")
    description: str | None = Field(
        default=None,
        description=(
            "Project summary — the explanatory text/bullet points written under the project "
            "entry on the resume, describing what it does or what was done."
        ),
    )
    technologies: list[str] = Field(
        default_factory=list, description="Languages, frameworks, or tools used in the project"
    )
    url: str | None = Field(default=None, description="Project link, e.g. GitHub or live demo URL")


class CertificationItem(BaseModel):
    name: str
    issuer: str | None = None
    issue_date: str | None = None
    expiry_date: str | None = None
    credential_id: str | None = None


class LanguageItem(BaseModel):
    language: str = Field(
        description=(
            "A spoken/written human language the person knows, e.g. English, Hindi, Spanish. "
            "Never put programming, scripting, or markup languages here (e.g. C, C++, Java, "
            "Python, SQL, HTML) — those belong in the resume's `skills` list instead."
        )
    )
    proficiency: str | None = Field(default=None, description="e.g. native, fluent, conversational")


class ParsedResumeData(BaseModel):
    """Structured extraction contract.

    `model_json_schema()` on this class is the schema handed to
    `AIService.extract_structured` — it is both the API response shape and
    the Claude structured-output contract, so the two can never drift apart.
    """

    contact: ContactInfo = Field(
        description="Header/contact-block details from the top of the resume."
    )
    summary: str = Field(
        description=(
            "A concise 2-4 sentence professional summary. Look for a heading such as "
            "'Summary', 'Professional Summary', 'Objective', or 'About Me' near the top of "
            "the resume; if none exists, write one from the person's experience/skills instead "
            "of leaving this empty."
        )
    )
    skills: list[str] = Field(
        default_factory=list,
        description=(
            "Every technical and professional skill listed anywhere on the resume — under "
            "headings like 'Skills', 'Technical Skills', 'Core Competencies', or 'Tools & "
            "Technologies'. Always include programming/query/markup languages here "
            "(e.g. C, C++, Java, Python, SQL, HTML), plus frameworks, libraries, and tools."
        ),
    )
    experience: list[ExperienceItem] = Field(
        default_factory=list,
        description=(
            "Work history entries from headings like 'Experience', 'Work Experience', "
            "'Employment History', 'Internship', or 'Professional Experience'."
        ),
    )
    education: list[EducationItem] = Field(default_factory=list)
    projects: list[ProjectItem] = Field(
        default_factory=list,
        description=(
            "Entries from headings like 'Projects', 'Personal Projects', or 'Academic "
            "Projects' — including course or self-directed projects, not just employer work."
        ),
    )
    certifications: list[CertificationItem] = Field(
        default_factory=list,
        description=(
            "Entries from headings like 'Certifications', 'Certificates', 'Licenses & "
            "Certifications', or 'Courses and Workshops'."
        ),
    )
    achievements: list[str] = Field(default_factory=list)
    languages: list[LanguageItem] = Field(
        default_factory=list,
        description=(
            "Spoken/written human languages only, from a heading such as 'Languages'. Do NOT "
            "include programming languages here even if they appear under a heading literally "
            "named 'Languages' — put those in `skills` instead."
        ),
    )


class ParsedResumeDataNarrative(BaseModel):
    """First half of the extraction contract — contact/summary/skills/achievements/languages.

    Split out from `ParsedResumeData` because handing a small local model (the
    default `AI_PROVIDER=ollama` setup) one giant 8-field/5-nested-type schema
    in a single call causes it to fill only 1-2 fields and leave the rest at
    their empty defaults — trivially schema-valid, but useless. Two smaller
    calls (this + `ParsedResumeDataStructured`) measurably fix that: see
    `_parse_resume`. `contact` lives here rather than in a third call since
    it's flat/simple fields, same rationale as summary/skills.
    """

    contact: ContactInfo = ParsedResumeData.model_fields["contact"]
    summary: str = ParsedResumeData.model_fields["summary"]
    skills: list[str] = ParsedResumeData.model_fields["skills"]
    achievements: list[str] = ParsedResumeData.model_fields["achievements"]
    languages: list[LanguageItem] = ParsedResumeData.model_fields["languages"]


class ParsedResumeDataStructured(BaseModel):
    """Second half of the extraction contract — experience/education/projects/
    certifications. See `ParsedResumeDataNarrative` for why this is split out.
    """

    experience: list[ExperienceItem] = ParsedResumeData.model_fields["experience"]
    education: list[EducationItem] = ParsedResumeData.model_fields["education"]
    projects: list[ProjectItem] = ParsedResumeData.model_fields["projects"]
    certifications: list[CertificationItem] = ParsedResumeData.model_fields["certifications"]


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
    "ContactInfo",
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
