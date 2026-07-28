"""Unit tests for `app.modules.ats_resume_generator.validation` — pure
functions, no DB/HTTP/AI I-O required.
"""
from __future__ import annotations

from app.modules.ats_resume_generator.content import ResumeContent
from app.modules.ats_resume_generator.validation import check_page_overflow, validate_resume_content

_FULL_CONTACT = {
    "full_name": "Jane Doe",
    "phone": "+1 555-0100",
    "email": "jane@example.com",
    "linkedin_url": "linkedin.com/in/janedoe",
    "github_url": "github.com/janedoe",
    "portfolio_url": "janedoe.dev",
    "location": "Remote",
}


def _content(**overrides) -> ResumeContent:
    defaults = dict(
        contact=_FULL_CONTACT,
        summary="Backend engineer with 5 years of experience.",
        grouped_skills=[{"category": "Languages", "skills": ["Python"]}],
        experience=[],
        projects=[],
        education=[],
        certifications=[],
        achievements=[],
        languages=[],
    )
    defaults.update(overrides)
    return ResumeContent(**defaults)


def test_valid_content_passes_with_no_errors_or_warnings() -> None:
    result = validate_resume_content(_content())
    assert result.passed
    assert result.errors == []
    assert result.warnings == []


def test_missing_name_is_a_hard_error() -> None:
    result = validate_resume_content(_content(contact={**_FULL_CONTACT, "full_name": ""}))
    assert not result.passed
    assert any("name" in e.lower() for e in result.errors)


def test_missing_email_and_phone_is_a_hard_error() -> None:
    contact = {**_FULL_CONTACT, "email": None, "phone": None}
    result = validate_resume_content(_content(contact=contact))
    assert not result.passed
    assert any("email" in e.lower() and "phone" in e.lower() for e in result.errors)


def test_having_only_phone_is_sufficient() -> None:
    contact = {**_FULL_CONTACT, "email": None}
    result = validate_resume_content(_content(contact=contact))
    assert result.passed


def test_missing_summary_is_a_hard_error() -> None:
    result = validate_resume_content(_content(summary=""))
    assert not result.passed
    assert any("summary" in e.lower() for e in result.errors)


def test_empty_resume_is_a_hard_error() -> None:
    result = validate_resume_content(_content(grouped_skills=[], summary="Something."))
    assert not result.passed
    assert any("nothing substantive" in e for e in result.errors)


def test_missing_links_are_warnings_not_errors() -> None:
    contact = {**_FULL_CONTACT, "linkedin_url": None, "github_url": None, "portfolio_url": None, "location": None}
    result = validate_resume_content(_content(contact=contact))
    assert result.passed
    assert len(result.warnings) == 4


def test_project_with_too_few_bullets_warns() -> None:
    projects = [{"name": "Order Service", "highlights": ["Only one bullet"]}]
    result = validate_resume_content(_content(projects=projects))
    assert result.passed
    assert any("Order Service" in w and "bullet" in w for w in result.warnings)


def test_project_with_too_many_bullets_warns() -> None:
    projects = [{"name": "Order Service", "highlights": [f"Bullet {i}" for i in range(6)]}]
    result = validate_resume_content(_content(projects=projects))
    assert result.passed
    assert any("trimming" in w for w in result.warnings)


def test_project_with_ideal_bullet_count_has_no_warning() -> None:
    projects = [{"name": "Order Service", "highlights": [f"Bullet {i}" for i in range(4)]}]
    result = validate_resume_content(_content(projects=projects))
    assert not any("Order Service" in w for w in result.warnings)


def test_check_page_overflow() -> None:
    assert check_page_overflow(1) is None
    assert check_page_overflow(2) is None
    assert check_page_overflow(3) is not None
