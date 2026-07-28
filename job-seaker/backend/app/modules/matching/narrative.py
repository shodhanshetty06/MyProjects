"""LLM-generated one-line "why this fits" narrative for a resume/job match.

Deliberately thin: takes only the already-computed, already-persisted match
data (`matched_skills`, `missing_skills`, `strengths`, `weaknesses`,
`match_score`) — never re-derives the score itself and never sees the raw
resume/job text, so a bad or unavailable AI provider can only ever cost the
caller a missing narrative, not a wrong score. Callers must treat a `None`
return as "not available" and continue without it (see
`job_feed_service.py::_get_or_generate_narrative`).
"""

from __future__ import annotations

from app.services.ai.service import AIService

NARRATIVE_SYSTEM_PROMPT = (
    "You are a career assistant summarizing why a candidate matches a job posting. "
    "You are given the candidate's matched skills, missing skills, and templated "
    "strengths/weaknesses notes for one specific match. Write exactly one sentence "
    "(max 220 characters) explaining why this match scored the way it did. "
    "Only reference the skills and notes provided — never invent a skill, employer, "
    "or qualification that isn't in the input. Be concrete and specific, not generic."
)


def _build_narrative_prompt(
    *,
    matched_skills: list[str],
    missing_skills: list[str],
    strengths: list[str],
    weaknesses: list[str],
    match_score: float | None,
) -> str:
    score_line = f"{match_score:.0f}%" if match_score is not None else "unknown"
    return (
        f"Match score: {score_line}\n"
        f"Matched skills: {', '.join(matched_skills) or '(none)'}\n"
        f"Missing skills: {', '.join(missing_skills) or '(none)'}\n"
        f"Strengths notes: {' '.join(strengths) or '(none)'}\n"
        f"Weaknesses notes: {' '.join(weaknesses) or '(none)'}"
    )


async def generate_match_narrative(
    *,
    ai_service: AIService,
    matched_skills: list[str],
    missing_skills: list[str],
    strengths: list[str],
    weaknesses: list[str],
    match_score: float | None,
) -> str | None:
    """Returns a one-sentence narrative, or `None` if there's nothing
    meaningful to summarize (no skills and no notes at all)."""
    if not matched_skills and not missing_skills and not strengths and not weaknesses:
        return None

    prompt = _build_narrative_prompt(
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        strengths=strengths,
        weaknesses=weaknesses,
        match_score=match_score,
    )
    narrative = await ai_service.complete_chat(
        system=NARRATIVE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
    )
    return narrative.strip() or None


__all__ = ["generate_match_narrative", "NARRATIVE_SYSTEM_PROMPT"]
