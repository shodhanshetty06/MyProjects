"""Application-matching heuristic — pure function, no I/O, fully unit-testable.

Scores each candidate `Application` against an incoming email on two signals:
sender-domain/extracted-company-name vs. `company_name`, and subject/snippet/
extracted-job-title vs. `job_title`. Uses only the stdlib (`difflib`), matching
the codebase's lightweight-dependency ethos — no fuzzy-matching package added.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.modules.tracking.models import Application

# Domains that are never meaningful to compare against a company name: ATS
# platforms, transactional-email senders, and generic consumer webmail.
ATS_ESP_DOMAIN_DENYLIST = frozenset(
    {
        "greenhouse.io",
        "lever.co",
        "myworkday.com",
        "smartrecruiters.com",
        "icims.com",
        "taleo.net",
        "successfactors.com",
        "jobvite.com",
        "workable.com",
        "bamboohr.com",
        "ashbyhq.com",
        "sendgrid.net",
        "amazonses.com",
        "mailgun.org",
        "gmail.com",
        "outlook.com",
        "hotmail.com",
        "yahoo.com",
    }
)

_COMPANY_SUFFIX_RE = re.compile(
    r"\b(inc|llc|corp|corporation|co|ltd|limited|group|technologies|technology|tech)\b\.?",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_EMAIL_DOMAIN_RE = re.compile(r"@([\w.-]+)")


@dataclass
class MatchResult:
    application: Application | None
    score: float
    is_confident: bool


def _normalize_company(name: str) -> str:
    return " ".join(_COMPANY_SUFFIX_RE.sub("", name.lower()).split())


def _sender_domain(sender: str) -> str | None:
    match = _EMAIL_DOMAIN_RE.search(sender)
    return match.group(1).lower() if match else None


def _company_score(*, sender: str, extracted_company_name: str | None, company_name: str) -> float:
    normalized_target = _normalize_company(company_name)
    if not normalized_target:
        return 0.0

    scores: list[float] = []

    domain = _sender_domain(sender)
    if domain and domain not in ATS_ESP_DOMAIN_DENYLIST:
        domain_label = domain.split(".")[0]
        scores.append(
            SequenceMatcher(None, domain_label, normalized_target.replace(" ", "")).ratio()
        )

    if extracted_company_name:
        scores.append(
            SequenceMatcher(
                None, _normalize_company(extracted_company_name), normalized_target
            ).ratio()
        )

    return max(scores) if scores else 0.0


def _tokenize(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text.lower()) if len(token) > 2}


def _title_score(
    *, subject: str, snippet: str, extracted_job_title: str | None, job_title: str
) -> float:
    target_tokens = _tokenize(job_title)
    if not target_tokens:
        return 0.0
    source_tokens = _tokenize(" ".join(filter(None, [subject, snippet, extracted_job_title])))
    if not source_tokens:
        return 0.0
    union = target_tokens | source_tokens
    return len(target_tokens & source_tokens) / len(union) if union else 0.0


def match_application(
    *,
    candidates: Sequence[Application],
    sender: str,
    subject: str,
    snippet: str,
    extracted_company_name: str | None,
    extracted_job_title: str | None,
    company_weight: float,
    title_weight: float,
    confidence_threshold: float,
    margin: float,
) -> MatchResult:
    """Score every candidate and return the best one.

    `is_confident` is true only when the top score clears `confidence_threshold`
    *and* beats the runner-up by at least `margin` — avoiding a confident pick
    between two similarly-named companies. Callers should treat a non-confident
    result with `application` still set as a "best guess" for manual review,
    and a `None` application (empty `candidates`) as fully unmatched.
    """
    if not candidates:
        return MatchResult(application=None, score=0.0, is_confident=False)

    scored = sorted(
        (
            (
                company_weight
                * _company_score(
                    sender=sender,
                    extracted_company_name=extracted_company_name,
                    company_name=candidate.company_name,
                )
                + title_weight
                * _title_score(
                    subject=subject,
                    snippet=snippet,
                    extracted_job_title=extracted_job_title,
                    job_title=candidate.job_title,
                ),
                candidate,
            )
            for candidate in candidates
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )

    best_score, best_candidate = scored[0]
    runner_up_score = scored[1][0] if len(scored) > 1 else 0.0
    is_confident = best_score >= confidence_threshold and (best_score - runner_up_score) >= margin

    return MatchResult(application=best_candidate, score=best_score, is_confident=is_confident)


__all__ = ["MatchResult", "match_application", "ATS_ESP_DOMAIN_DENYLIST"]
