"""AI-based email classification — reuses `AIService.extract_structured`
(`app/services/ai/service.py`) rather than a second provider client, same
convention `job_ingestion/tasks.py::_extract_job_skills` follows.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.services.ai.service import AIService

EMAIL_CLASSIFICATION_SYSTEM_PROMPT = (
    "You are an assistant that reviews an email received by a job seeker and determines "
    "whether it relates to the status of a job application they submitted. Classify the "
    "email into exactly one of: 'interview' (an interview is being scheduled or discussed), "
    "'assessment' (a coding challenge, test, or take-home assignment is being requested), "
    "'rejection' (the candidate was not selected), 'offer' (a job offer was extended), or "
    "'none' (the email is not related to a specific job application's status — e.g. a job "
    "alert, newsletter, marketing email, or unrelated personal/professional email). "
    "Set is_relevant=false whenever classification is 'none'. Extract the hiring company's "
    "name and the job title if you can infer them from the email, otherwise leave them null. "
    "Provide a confidence score between 0 and 1 reflecting how certain you are, and a short "
    "reasoning sentence."
)


class EmailClassificationResult(BaseModel):
    is_relevant: bool
    classification: Literal["interview", "assessment", "rejection", "offer", "none"]
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_company_name: str | None = None
    extracted_job_title: str | None = None
    reasoning: str


async def classify_email(
    ai_service: AIService, *, sender: str, subject: str, snippet: str
) -> EmailClassificationResult:
    text = f"From: {sender}\nSubject: {subject}\n\n{snippet}"
    raw = await ai_service.extract_structured(
        text=text,
        schema=EmailClassificationResult.model_json_schema(),
        system=EMAIL_CLASSIFICATION_SYSTEM_PROMPT,
    )
    return EmailClassificationResult.model_validate(raw)


__all__ = ["EmailClassificationResult", "EMAIL_CLASSIFICATION_SYSTEM_PROMPT", "classify_email"]
