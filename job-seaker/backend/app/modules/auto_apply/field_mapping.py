"""Field Mapping — resolves an ATS's discovered `ApplicationField`s against
an applicant's data, and reports anything that couldn't be resolved.

Two-pass resolution per field:
    1. Canonical match — the adapter already told us which profile field a
       form field is (`ApplicationField.applicant_field_hint`, e.g.
       Greenhouse's `#first_name`), or the field's label matches one of the
       keyword hints in `CANONICAL_FIELD_HINTS` (e.g. a Lever question
       labeled "Are you legally authorized to work?").
    2. Custom-answer match — the profile's free-form `custom_answers` bank
       (question text -> answer), matched by normalized label. This is how
       a user "teaches" the engine answers to company-specific screening
       questions over time.

Deliberately pure/DB-free: `map_fields()` takes plain dicts so it's cheaply
unit testable. `profile_to_field_values()` is the one function that touches
the `ApplicantProfile` ORM row, converting it into the flat dict `map_fields`
expects.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.modules.auto_apply.adapters.base import ApplicationField, FieldType
from app.modules.auto_apply.models import ApplicantProfile

# canonical_field_name -> label keywords that identify it on an arbitrary ATS form.
CANONICAL_FIELD_HINTS: dict[str, list[str]] = {
    "first_name": ["first name", "given name"],
    "last_name": ["last name", "family name", "surname"],
    "full_name": ["full name", "your name"],
    "email": ["email", "e-mail"],
    "phone": ["phone", "mobile", "telephone"],
    "address_line1": ["address", "street address"],
    "city": ["city"],
    "state": ["state", "province"],
    "postal_code": ["postal code", "zip code", "zip"],
    "country": ["country", "location"],
    "linkedin_url": ["linkedin"],
    "portfolio_url": ["portfolio", "personal website"],
    "github_url": ["github"],
    "website_url": ["website"],
    "work_authorized": ["authorized to work", "legally authorized", "work authorization"],
    "requires_sponsorship": ["sponsorship", "visa"],
    "willing_to_relocate": ["relocate", "relocation"],
    "years_of_experience": ["years of experience", "years experience"],
    "desired_salary": ["desired salary", "salary expectation", "compensation expectation"],
    "notice_period": ["notice period"],
    "gender": ["gender"],
    "race_ethnicity": ["race", "ethnicity"],
    "veteran_status": ["veteran"],
    "disability_status": ["disability"],
}

_PUNCTUATION_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")


class FieldMappingResult(BaseModel):
    mapped_values: dict[str, str] = Field(default_factory=dict)
    missing_fields: list[ApplicationField] = Field(default_factory=list)
    resolved_count: int = 0
    total_required_count: int = 0

    @property
    def is_complete(self) -> bool:
        return not self.missing_fields


def normalize_label(label: str) -> str:
    lowered = label.strip().lower().rstrip("*").strip()
    stripped = _PUNCTUATION_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", stripped).strip()


def profile_to_field_values(profile: ApplicantProfile) -> dict[str, str]:
    """Flatten an `ApplicantProfile` row into canonical field_name -> value strings."""
    full_name = " ".join(part for part in (profile.first_name, profile.last_name) if part)
    raw_values: dict[str, object] = {
        "first_name": profile.first_name,
        "last_name": profile.last_name,
        "full_name": full_name or None,
        "email": profile.email,
        "phone": profile.phone,
        "address_line1": profile.address_line1,
        "city": profile.city,
        "state": profile.state,
        "postal_code": profile.postal_code,
        "country": profile.country,
        "linkedin_url": profile.linkedin_url,
        "portfolio_url": profile.portfolio_url,
        "github_url": profile.github_url,
        "website_url": profile.website_url,
        "work_authorized": profile.work_authorized,
        "requires_sponsorship": profile.requires_sponsorship,
        "willing_to_relocate": profile.willing_to_relocate,
        "years_of_experience": profile.years_of_experience,
        "desired_salary": profile.desired_salary,
        "notice_period": profile.notice_period,
        "gender": profile.gender,
        "race_ethnicity": profile.race_ethnicity,
        "veteran_status": profile.veteran_status,
        "disability_status": profile.disability_status,
    }
    return {
        key: ("yes" if value is True else "no" if value is False else str(value))
        for key, value in raw_values.items()
        if value is not None
    }


def _resolve_canonical(field: ApplicationField, field_values: dict[str, str]) -> str | None:
    if field.applicant_field_hint and field.applicant_field_hint in field_values:
        return field_values[field.applicant_field_hint]

    normalized_label = normalize_label(field.label)
    for canonical_key, keywords in CANONICAL_FIELD_HINTS.items():
        if canonical_key not in field_values:
            continue
        if any(keyword in normalized_label for keyword in keywords):
            return field_values[canonical_key]
    return None


def _resolve_custom_answer(field: ApplicationField, custom_answers: dict[str, str]) -> str | None:
    normalized_label = normalize_label(field.label)
    normalized_answers = {normalize_label(question): answer for question, answer in custom_answers.items()}

    if normalized_label in normalized_answers:
        return normalized_answers[normalized_label]

    for question, answer in normalized_answers.items():
        if question and (question in normalized_label or normalized_label in question):
            return answer
    return None


def map_fields(
    fields: list[ApplicationField],
    *,
    field_values: dict[str, str],
    custom_answers: dict[str, str] | None = None,
) -> FieldMappingResult:
    """Resolve every discovered field against `field_values`/`custom_answers`.

    File fields (resume upload) are intentionally excluded from mapping —
    the resolved resume attachment is handled directly by the adapter, not
    through this key/value flow.
    """
    custom_answers = custom_answers or {}
    mapped: dict[str, str] = {}
    missing: list[ApplicationField] = []
    total_required = 0

    for field in fields:
        if field.field_type == FieldType.FILE:
            continue
        if field.required:
            total_required += 1

        value = _resolve_canonical(field, field_values)
        if value is None:
            value = _resolve_custom_answer(field, custom_answers)

        if value is not None:
            mapped[field.key] = value
        elif field.required:
            missing.append(field)

    return FieldMappingResult(
        mapped_values=mapped,
        missing_fields=missing,
        resolved_count=len(mapped),
        total_required_count=total_required,
    )


__all__ = [
    "CANONICAL_FIELD_HINTS",
    "FieldMappingResult",
    "normalize_label",
    "profile_to_field_values",
    "map_fields",
]
