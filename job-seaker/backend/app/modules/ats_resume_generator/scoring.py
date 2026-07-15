"""Pure scoring logic for the ATS Resume Generator module.

Deliberately free of DB/HTTP/AI I-O so it's unit-testable in isolation,
mirroring `app.modules.matching.scoring`. Callers (the service/task layer)
are responsible for extracting plain values from ORM objects / AI responses
before calling in.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_QUANTIFIED_RE = re.compile(r"\d")


def _normalize(text: str) -> str:
    return text.strip().lower()


@dataclass(frozen=True)
class KeywordCoverage:
    matched: list[str]
    missing: list[str]
    score: float  # 0..100


def keyword_coverage(keywords: Sequence[str], resume_text: str) -> KeywordCoverage:
    """How many of `keywords` (JD-derived terms) appear in `resume_text`.

    A keyword "matches" if it appears as a substring of the (lowercased)
    resume text — good enough for short technical terms/phrases without a
    full NLP stemmer, and tolerant of simple pluralization ("APIs" still
    contains "api").

    A job description that yielded no keywords scores 100 on this dimension
    rather than 0 — nothing to be missing when there's nothing to look for.
    """
    haystack = _normalize(resume_text)
    # Dedupe case-insensitively (keeping first-seen casing) so "Python" and
    # "python" from an AI-generated keyword list don't get double-counted.
    seen: dict[str, str] = {}
    for keyword in keywords:
        stripped = keyword.strip()
        if stripped and _normalize(stripped) not in seen:
            seen[_normalize(stripped)] = stripped
    unique_keywords = list(seen.values())
    if not unique_keywords:
        return KeywordCoverage(matched=[], missing=[], score=100.0)

    matched = [k for k in unique_keywords if _normalize(k) in haystack]
    missing = [k for k in unique_keywords if _normalize(k) not in haystack]
    score = (len(matched) / len(unique_keywords)) * 100.0
    return KeywordCoverage(matched=matched, missing=missing, score=score)


def build_resume_search_text(
    *,
    summary: str,
    skills: Sequence[str],
    projects: Sequence[dict],
    experience: Sequence[dict],
    education: Sequence[dict],
    certifications: Sequence[dict],
) -> str:
    """Flatten every resume section into one blob for keyword search."""
    parts: list[str] = [summary, " ".join(skills)]
    for project in projects:
        parts.append(project.get("name", "") or "")
        parts.append(project.get("description", "") or "")
        parts.extend(project.get("technologies", []) or [])
        parts.extend(project.get("highlights", []) or [])
    for job in experience:
        parts.append(job.get("title", "") or "")
        parts.append(job.get("description", "") or "")
        parts.extend(job.get("highlights", []) or [])
    for edu in education:
        parts.append(edu.get("degree", "") or "")
        parts.append(edu.get("field_of_study", "") or "")
    for cert in certifications:
        parts.append(cert.get("name", "") or "")
    return " \n ".join(p for p in parts if p)


def _quantified_ratio(bullets: Iterable[str]) -> float:
    bullets = list(bullets)
    if not bullets:
        return 0.0
    quantified = sum(1 for b in bullets if _QUANTIFIED_RE.search(b))
    return quantified / len(bullets)


@dataclass(frozen=True)
class StructureScore:
    score: float  # 0..100
    checks: dict[str, bool | float]


def structure_score(
    *,
    has_summary: bool,
    has_skills: bool,
    has_experience: bool,
    has_education: bool,
    has_projects: bool,
    bullet_highlights: Sequence[str],
) -> StructureScore:
    """Structural/formatting signal independent of keyword coverage — the
    standard-sections-present + quantified-impact heuristics ATS parsers and
    human reviewers both reward. Scored out of 100:
        - 15 pts each for a summary / skills / experience / education
          section being present (60 total)
        - 15 pts for a projects section being present
        - up to 25 pts scaled by the fraction of bullets that cite a number
          ("led a team of 5", "cut latency by 30%") — a common signal of
          concrete, quantified impact rather than vague duty statements.
    """
    quantified_ratio = _quantified_ratio(bullet_highlights)
    checks: dict[str, bool | float] = {
        "has_summary": has_summary,
        "has_skills": has_skills,
        "has_experience": has_experience,
        "has_education": has_education,
        "has_projects": has_projects,
        "quantified_bullet_ratio": round(quantified_ratio, 2),
    }
    section_score = sum(
        15.0 for present in (has_summary, has_skills, has_experience, has_education) if present
    ) + (15.0 if has_projects else 0.0)
    quantified_score = quantified_ratio * 25.0
    return StructureScore(score=min(100.0, section_score + quantified_score), checks=checks)


def compute_ats_score(
    *,
    keyword_score: float,
    structure_score_value: float,
    keyword_weight: float,
    structure_weight: float,
) -> float:
    """Combine keyword coverage + structural signal into a 0..100 score."""
    total = keyword_score * keyword_weight + structure_score_value * structure_weight
    return max(0.0, min(100.0, total))


__all__ = [
    "KeywordCoverage",
    "keyword_coverage",
    "build_resume_search_text",
    "StructureScore",
    "structure_score",
    "compute_ats_score",
]
