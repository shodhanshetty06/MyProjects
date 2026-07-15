"""Unit tests for Confirmation Detection."""
from __future__ import annotations

from app.modules.auto_apply.confirmation import detect_confirmation


def test_detects_confirmation_phrase_in_html() -> None:
    result = detect_confirmation(
        text="<html><body><h1>Thank you for applying!</h1></body></html>", url=None
    )
    assert result.detected is True
    assert result.matched_via == "text"


def test_detects_confirmation_via_url_pattern_when_no_phrase() -> None:
    result = detect_confirmation(text="<html><body>OK</body></html>", url="https://boards.greenhouse.io/acme/confirmation")
    assert result.detected is True
    assert result.matched_via == "url"


def test_text_match_takes_priority_over_url() -> None:
    result = detect_confirmation(
        text="Application received", url="https://example.com/jobs/123"
    )
    assert result.detected is True
    assert result.matched_via == "text"


def test_no_confirmation_detected_for_unrelated_page() -> None:
    result = detect_confirmation(text="<html><body>Please fix the errors below.</body></html>", url=None)
    assert result.detected is False
    assert result.matched_phrase is None


def test_handles_empty_input() -> None:
    result = detect_confirmation(text=None, url=None)
    assert result.detected is False
