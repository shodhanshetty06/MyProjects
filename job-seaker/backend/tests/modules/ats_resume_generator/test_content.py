"""Unit tests for `app.modules.ats_resume_generator.content` — pure function
over an `ATSResumeVersion`-shaped object, no DB required.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.modules.ats_resume_generator.content import build_resume_content


def _fake_version(**overrides):
    defaults = dict(
        contact={"full_name": "Jane Doe", "email": "jane@example.com"},
        optimized_summary="Backend engineer.",
        grouped_skills=[{"category": "Languages", "skills": ["Python"]}],
        optimized_skills=["Python"],
        experience=[{"title": "Engineer", "company": "Acme"}],
        optimized_projects=[{"name": "Order Service"}],
        education=[{"institution": "State University"}],
        certifications=[{"name": "AWS Certified"}],
        achievements=["Speaker at PyCon"],
        languages=[{"language": "English"}],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_build_resume_content_maps_all_sections() -> None:
    content = build_resume_content(_fake_version())

    assert content.contact["full_name"] == "Jane Doe"
    assert content.summary == "Backend engineer."
    assert content.grouped_skills == [{"category": "Languages", "skills": ["Python"]}]
    assert content.experience == [{"title": "Engineer", "company": "Acme"}]
    assert content.projects == [{"name": "Order Service"}]
    assert content.education == [{"institution": "State University"}]
    assert content.certifications == [{"name": "AWS Certified"}]
    assert content.achievements == ["Speaker at PyCon"]
    assert content.languages == [{"language": "English"}]


def test_build_resume_content_falls_back_to_flat_skills_when_ungrouped() -> None:
    version = _fake_version(grouped_skills=[], optimized_skills=["Python", "SQL"])
    content = build_resume_content(version)

    assert content.grouped_skills == [{"category": "Skills", "skills": ["Python", "SQL"]}]


def test_build_resume_content_handles_none_collections() -> None:
    version = _fake_version(
        experience=None,
        optimized_projects=None,
        education=None,
        certifications=None,
        achievements=None,
        languages=None,
        contact=None,
    )
    content = build_resume_content(version)

    assert content.contact == {}
    assert content.experience == []
    assert content.projects == []
    assert content.education == []
    assert content.certifications == []
    assert content.achievements == []
    assert content.languages == []
