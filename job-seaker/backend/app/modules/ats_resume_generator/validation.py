"""ATS Resume Generator module — pre-export validation.

Pure function over a `ResumeContent`, no DB/HTTP/AI I-O, mirroring
`scoring.py`. `errors` block export entirely (`_generate_ats_resume_files`
marks the version FAILED and generates no files); `warnings` are stored on
the version and surfaced in the UI but don't block generation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.modules.ats_resume_generator.content import ResumeContent

_MAX_BULLET_CHARS = 220
_MIN_PROJECT_BULLETS = 3
_MAX_PROJECT_BULLETS = 5
_MAX_PAGES_BEFORE_OVERFLOW_WARNING = 2


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_resume_content(content: ResumeContent) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    full_name = (content.contact.get("full_name") or "").strip()
    email = (content.contact.get("email") or "").strip()
    phone = (content.contact.get("phone") or "").strip()

    if not full_name:
        errors.append("Missing candidate name in the contact header.")
    if not email and not phone:
        errors.append("Missing both email and phone — at least one reachable contact method is required.")

    if not (content.summary or "").strip():
        errors.append("Missing professional summary.")

    has_skills = any((group.get("skills") or []) for group in content.grouped_skills)
    if not has_skills and not content.experience and not content.projects:
        errors.append("Resume has no skills, experience, or projects — nothing substantive to export.")

    if not content.contact.get("linkedin_url"):
        warnings.append("No LinkedIn URL found.")
    if not content.contact.get("github_url"):
        warnings.append("No GitHub URL found.")
    if not content.contact.get("portfolio_url"):
        warnings.append("No portfolio URL found.")
    if not content.contact.get("location"):
        warnings.append("No location found.")

    for project in content.projects:
        name = project.get("name") or "Untitled project"
        highlights = project.get("highlights") or []
        if len(highlights) < _MIN_PROJECT_BULLETS:
            warnings.append(
                f"Project '{name}' has only {len(highlights)} bullet point(s) — aim for "
                f"{_MIN_PROJECT_BULLETS}-{_MAX_PROJECT_BULLETS}."
            )
        elif len(highlights) > _MAX_PROJECT_BULLETS:
            warnings.append(
                f"Project '{name}' has {len(highlights)} bullet points — consider trimming to "
                f"{_MIN_PROJECT_BULLETS}-{_MAX_PROJECT_BULLETS}."
            )
        for bullet in highlights:
            if len(bullet) > _MAX_BULLET_CHARS:
                warnings.append(f"A bullet under '{name}' is very long and may wrap poorly.")
                break

    for job in content.experience:
        for bullet in job.get("highlights") or []:
            if len(bullet) > _MAX_BULLET_CHARS:
                title = job.get("title") or job.get("company") or "an experience entry"
                warnings.append(f"A bullet under '{title}' is very long and may wrap poorly.")
                break

    return ValidationResult(passed=not errors, errors=errors, warnings=warnings)


def check_page_overflow(page_count: int) -> str | None:
    if page_count > _MAX_PAGES_BEFORE_OVERFLOW_WARNING:
        return f"Generated document is {page_count} pages — consider trimming content to fit {_MAX_PAGES_BEFORE_OVERFLOW_WARNING}."
    return None


__all__ = ["ValidationResult", "validate_resume_content", "check_page_overflow"]
