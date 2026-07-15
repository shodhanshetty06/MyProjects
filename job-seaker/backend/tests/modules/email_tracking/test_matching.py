"""Unit tests for the application-matching heuristic — pure functions, no
DB/network access required. `Application` instances are constructed directly
(never persisted) since `match_application` only reads `company_name`/
`job_title` off them.
"""

from __future__ import annotations

from app.modules.email_tracking.matching import match_application
from app.modules.tracking.models import Application

_WEIGHTS = {
    "company_weight": 0.7,
    "title_weight": 0.3,
    "confidence_threshold": 0.72,
    "margin": 0.15,
}


def _application(company_name: str, job_title: str) -> Application:
    return Application(company_name=company_name, job_title=job_title)


def test_confident_match_on_strong_company_and_title_signal() -> None:
    candidates = [
        _application("Acme Corp", "Backend Engineer"),
        _application("Globex Inc", "Frontend Engineer"),
    ]

    result = match_application(
        candidates=candidates,
        sender="recruiting@acme.com",
        subject="Your interview with Acme",
        snippet="We'd like to schedule an interview for the Backend Engineer role.",
        extracted_company_name="Acme",
        extracted_job_title="Backend Engineer",
        **_WEIGHTS,
    )

    assert result.is_confident is True
    assert result.application is candidates[0]


def test_ambiguous_candidates_are_not_confident() -> None:
    # "Corp"/"Inc" are both stripped by the company-suffix normalizer, so
    # these two candidates score identically — a genuinely ambiguous pick
    # the margin check should refuse to resolve confidently.
    candidates = [
        _application("Acme Corp", "Backend Engineer"),
        _application("Acme Inc", "Backend Engineer"),
    ]

    result = match_application(
        candidates=candidates,
        sender="noreply@acme.com",
        subject="Update on your application",
        snippet="Thanks for applying to the Backend Engineer role.",
        extracted_company_name="Acme",
        extracted_job_title="Backend Engineer",
        **_WEIGHTS,
    )

    assert result.is_confident is False
    assert result.application is not None  # best guess is still attached


def test_no_candidates_is_unmatched() -> None:
    result = match_application(
        candidates=[],
        sender="recruiting@acme.com",
        subject="Your interview with Acme",
        snippet="Let's schedule your interview.",
        extracted_company_name="Acme",
        extracted_job_title="Backend Engineer",
        **_WEIGHTS,
    )

    assert result.application is None
    assert result.is_confident is False
    assert result.score == 0.0


def test_ats_esp_domain_is_not_scored_against_company_name() -> None:
    # A Greenhouse-hosted email should not score highly against "Acme Corp"
    # purely because the domain string happens to share letters with it —
    # the denylist should force reliance on the extracted_company_name signal.
    candidates = [_application("Acme Corp", "Backend Engineer")]

    result_without_extraction = match_application(
        candidates=candidates,
        sender="no-reply@greenhouse.io",
        subject="Application received",
        snippet="Thanks for applying.",
        extracted_company_name=None,
        extracted_job_title=None,
        **_WEIGHTS,
    )

    assert result_without_extraction.score == 0.0


def test_low_score_candidate_is_still_returned_as_best_guess() -> None:
    candidates = [_application("Totally Unrelated Co", "Marketing Manager")]

    result = match_application(
        candidates=candidates,
        sender="hr@somewhere-else.com",
        subject="Interview scheduling",
        snippet="Let's find a time to talk.",
        extracted_company_name=None,
        extracted_job_title=None,
        **_WEIGHTS,
    )

    assert result.is_confident is False
    assert result.application is candidates[0]
    assert result.score < 0.3
