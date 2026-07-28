"""Unit tests for `app.modules.ats_resume_generator.templates` — pure data,
no DB/HTTP/AI I-O required.
"""
from __future__ import annotations

from app.modules.ats_resume_generator.templates import (
    DEFAULT_TEMPLATE_ID,
    TEMPLATES,
    get_template,
    list_templates,
)

_EXPECTED_IDS = {"modern_ats", "executive", "minimal", "professional", "tech", "consulting"}


def test_all_six_premium_templates_are_registered() -> None:
    assert set(TEMPLATES.keys()) == _EXPECTED_IDS
    assert len(list_templates()) == 6


def test_default_template_id_is_registered() -> None:
    assert DEFAULT_TEMPLATE_ID in TEMPLATES


def test_get_template_falls_back_to_default_for_unknown_id() -> None:
    assert get_template("does-not-exist") is TEMPLATES[DEFAULT_TEMPLATE_ID]
    assert get_template(None) is TEMPLATES[DEFAULT_TEMPLATE_ID]


def test_get_template_returns_requested_template() -> None:
    assert get_template("executive") is TEMPLATES["executive"]


def test_minimal_template_has_no_accent_color() -> None:
    assert TEMPLATES["minimal"].accent_color is None
