"""Confirmation Detection — pure heuristics for recognizing a successful
application submission from response text/HTML and the landing URL.

Deliberately has no I/O so it can be unit tested with plain strings and
reused identically by the HTTP-based adapters (Greenhouse, Lever) and the
Playwright-based ones (Workday, generic fallback).
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup
from pydantic import BaseModel

_CONFIRMATION_PHRASES = (
    "thank you for applying",
    "thanks for applying",
    "application received",
    "application submitted",
    "successfully submitted",
    "we've received your application",
    "we have received your application",
    "your application has been submitted",
    "your application was submitted",
)

_CONFIRMATION_URL_PATTERNS = (
    re.compile(r"/confirmation", re.IGNORECASE),
    re.compile(r"/thank-?you", re.IGNORECASE),
    re.compile(r"[?&]applied=true", re.IGNORECASE),
    re.compile(r"/application-confirmation", re.IGNORECASE),
)


class ConfirmationResult(BaseModel):
    detected: bool
    matched_phrase: str | None = None
    matched_via: str | None = None  # "text" | "url" | None
    confidence: float = 0.0


def _extract_text(html_or_text: str) -> str:
    stripped = html_or_text.strip()
    if not stripped:
        return ""
    if "<" in stripped and ">" in stripped:
        return BeautifulSoup(stripped, "lxml").get_text(separator=" ", strip=True)
    return stripped


def detect_confirmation(*, text: str | None = None, url: str | None = None) -> ConfirmationResult:
    """Best-effort check that a submission actually succeeded.

    Text match takes priority (higher confidence — an explicit confirmation
    phrase is stronger evidence than a URL shape), falling back to URL
    pattern matching when no phrase is found.
    """
    normalized_text = _extract_text(text or "").lower()
    for phrase in _CONFIRMATION_PHRASES:
        if phrase in normalized_text:
            return ConfirmationResult(
                detected=True, matched_phrase=phrase, matched_via="text", confidence=0.95
            )

    if url:
        for pattern in _CONFIRMATION_URL_PATTERNS:
            if pattern.search(url):
                return ConfirmationResult(
                    detected=True, matched_phrase=pattern.pattern, matched_via="url", confidence=0.6
                )

    return ConfirmationResult(detected=False)


__all__ = ["ConfirmationResult", "detect_confirmation"]
