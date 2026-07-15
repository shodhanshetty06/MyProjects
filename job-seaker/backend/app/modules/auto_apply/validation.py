"""Validation — checks that *resolved* field values are actually well-formed
for their field type, distinct from `field_mapping.py`'s job of finding a
value at all.

A field can be "mapped" (some value was found) yet still invalid — e.g. the
applicant's profile email is malformed, or a select field's mapped value
isn't one of the ATS's own option labels. Both cases should block
submission the same way a missing required field does, so `tasks.py` merges
`ValidationIssue`s into the same review workflow as missing fields.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from app.modules.auto_apply.adapters.base import ApplicationField, FieldType
from app.modules.auto_apply.field_mapping import normalize_label

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_DIGITS_RE = re.compile(r"\d")
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_MIN_PHONE_DIGITS = 7


class ValidationIssue(BaseModel):
    field_key: str
    label: str
    message: str


def _validate_value(field: ApplicationField, value: str) -> str | None:
    if field.field_type == FieldType.EMAIL and not _EMAIL_RE.match(value):
        return "Value does not look like a valid email address."
    if field.field_type == FieldType.PHONE and len(_PHONE_DIGITS_RE.findall(value)) < _MIN_PHONE_DIGITS:
        return "Value does not look like a valid phone number."
    if field.field_type == FieldType.URL and not _URL_RE.match(value):
        return "Value does not look like a valid URL."
    if field.field_type in (FieldType.SELECT, FieldType.RADIO) and field.options:
        allowed = {normalize_label(option) for option in field.options}
        if normalize_label(value) not in allowed:
            return f"'{value}' is not one of the allowed options for this field."
    return None


def validate_field_values(
    fields: list[ApplicationField], mapped_values: dict[str, str]
) -> list[ValidationIssue]:
    fields_by_key = {field.key: field for field in fields}
    issues: list[ValidationIssue] = []

    for key, value in mapped_values.items():
        field = fields_by_key.get(key)
        if field is None or not value:
            continue
        message = _validate_value(field, value)
        if message:
            issues.append(ValidationIssue(field_key=key, label=field.label, message=message))

    return issues


__all__ = ["ValidationIssue", "validate_field_values"]
