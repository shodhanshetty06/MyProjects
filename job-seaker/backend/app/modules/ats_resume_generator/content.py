"""ATS Resume Generator module — structured content model for rendering.

`ResumeContent` fields are declared in the exact section order the
generated resume must follow: header/contact, summary, skills (grouped),
experience, projects, education, certifications, achievements, languages.
`build_resume_content` is a pure function over an `ATSResumeVersion` so it's
trivially unit-testable without a DB or S3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.modules.ats_resume_generator.models import ATSResumeVersion


@dataclass
class ResumeContent:
    contact: dict
    summary: str
    grouped_skills: list[dict] = field(default_factory=list)
    experience: list[dict] = field(default_factory=list)
    projects: list[dict] = field(default_factory=list)
    education: list[dict] = field(default_factory=list)
    certifications: list[dict] = field(default_factory=list)
    achievements: list[str] = field(default_factory=list)
    languages: list[dict] = field(default_factory=list)


def build_resume_content(version: "ATSResumeVersion") -> ResumeContent:
    grouped_skills = list(version.grouped_skills or [])
    if not grouped_skills and version.optimized_skills:
        # Fallback for versions created before skill grouping existed.
        grouped_skills = [{"category": "Skills", "skills": list(version.optimized_skills)}]

    return ResumeContent(
        contact=dict(version.contact or {}),
        summary=version.optimized_summary,
        grouped_skills=grouped_skills,
        experience=list(version.experience or []),
        projects=list(version.optimized_projects or []),
        education=list(version.education or []),
        certifications=list(version.certifications or []),
        achievements=list(version.achievements or []),
        languages=list(version.languages or []),
    )


__all__ = ["ResumeContent", "build_resume_content"]
