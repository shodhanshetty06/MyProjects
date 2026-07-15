"""Unit tests for Field Mapping / Validation / Missing Fields."""
from __future__ import annotations

import uuid

from app.modules.auto_apply.adapters.base import ApplicationField, FieldType
from app.modules.auto_apply.field_mapping import map_fields, normalize_label, profile_to_field_values
from app.modules.auto_apply.models import ApplicantProfile
from app.modules.auto_apply.validation import validate_field_values


def _profile(**overrides) -> ApplicantProfile:
    defaults = dict(
        user_id=uuid.uuid4(),
        first_name="Ada",
        last_name="Lovelace",
        email="ada@example.com",
        phone="+1 415 555 0100",
        custom_answers={},
    )
    defaults.update(overrides)
    return ApplicantProfile(**defaults)


def test_normalize_label_strips_punctuation_and_asterisks() -> None:
    assert normalize_label("First Name *") == "first name"
    assert normalize_label("  E-Mail Address!! ") == "e mail address"


def test_map_fields_resolves_by_applicant_field_hint() -> None:
    profile = _profile()
    fields = [
        ApplicationField(key="first_name", label="First Name", required=True, applicant_field_hint="first_name"),
    ]
    result = map_fields(fields, field_values=profile_to_field_values(profile))
    assert result.mapped_values == {"first_name": "Ada"}
    assert result.is_complete


def test_map_fields_resolves_by_label_keyword_when_no_hint() -> None:
    profile = _profile()
    fields = [ApplicationField(key="q_email", label="Your Email Address", required=True)]
    result = map_fields(fields, field_values=profile_to_field_values(profile))
    assert result.mapped_values == {"q_email": "ada@example.com"}


def test_map_fields_falls_back_to_custom_answers() -> None:
    profile = _profile(custom_answers={"Why do you want to work here?": "Because I love the mission."})
    fields = [
        ApplicationField(key="q1", label="Why do you want to work here?", required=True),
    ]
    result = map_fields(
        fields, field_values=profile_to_field_values(profile), custom_answers=profile.custom_answers
    )
    assert result.mapped_values["q1"] == "Because I love the mission."


def test_map_fields_reports_missing_required_fields() -> None:
    profile = _profile(first_name=None)
    fields = [
        ApplicationField(key="first_name", label="First Name", required=True, applicant_field_hint="first_name"),
        ApplicationField(key="optional_q", label="Referral source", required=False),
    ]
    result = map_fields(fields, field_values=profile_to_field_values(profile))
    assert not result.is_complete
    assert [f.key for f in result.missing_fields] == ["first_name"]
    assert result.total_required_count == 1


def test_map_fields_skips_file_fields() -> None:
    profile = _profile()
    fields = [ApplicationField(key="resume", label="Resume", required=True, field_type=FieldType.FILE)]
    result = map_fields(fields, field_values=profile_to_field_values(profile))
    assert result.is_complete
    assert result.mapped_values == {}


def test_validate_field_values_flags_malformed_email() -> None:
    fields = [ApplicationField(key="email", label="Email", field_type=FieldType.EMAIL, required=True)]
    issues = validate_field_values(fields, {"email": "not-an-email"})
    assert len(issues) == 1
    assert issues[0].field_key == "email"


def test_validate_field_values_flags_value_outside_select_options() -> None:
    fields = [
        ApplicationField(
            key="visa", label="Visa status", field_type=FieldType.SELECT, options=["Yes", "No"], required=True
        )
    ]
    issues = validate_field_values(fields, {"visa": "Maybe"})
    assert len(issues) == 1

    ok_issues = validate_field_values(fields, {"visa": "yes"})
    assert ok_issues == []
