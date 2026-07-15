"""Shared plain-HTML `<form>` parsing/building helpers.

Greenhouse and Lever hosted apply pages are both, at their core, a single
server-rendered `<form>` — no SPA framework required to fill and submit
them — so `GreenhouseApplyAdapter` and `LeverApplyAdapter` share this parser
instead of each re-implementing "walk the DOM, infer a field type, find its
label" from scratch. Workday and the generic fallback are real SPAs and use
`browser.py` (Playwright) instead.
"""
from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from app.modules.auto_apply.adapters.base import ApplicationField, FieldType, ResumeAttachment

_TYPE_MAP = {
    "email": FieldType.EMAIL,
    "tel": FieldType.PHONE,
    "url": FieldType.URL,
    "file": FieldType.FILE,
    "date": FieldType.DATE,
    "checkbox": FieldType.CHECKBOX,
    "radio": FieldType.RADIO,
}


def _label_for(form: Tag, field: Tag) -> str:
    field_id = field.get("id")
    if field_id:
        label_el = form.find("label", attrs={"for": field_id})
        if label_el is not None:
            return label_el.get_text(strip=True)
    parent_label = field.find_parent("label")
    if parent_label is not None:
        return parent_label.get_text(strip=True)
    aria_label = field.get("aria-label")
    if aria_label:
        return str(aria_label)
    return str(field.get("name") or field.get("id") or "Untitled field")


def _is_required(field: Tag) -> bool:
    return field.has_attr("required") or field.get("aria-required") == "true"


def find_form(html: str, *, form_selector: str | None = None) -> Tag | None:
    soup = BeautifulSoup(html, "lxml")
    if form_selector:
        return soup.select_one(form_selector)
    return soup.find("form")


def parse_form_fields(form: Tag) -> list[ApplicationField]:
    """Extract every fillable `<input>`/`<textarea>`/`<select>` in `form`."""
    fields: list[ApplicationField] = []

    for element in form.find_all(["input", "textarea", "select"]):
        name = element.get("name")
        if not name:
            continue
        input_type = str(element.get("type", "text")).lower()
        if input_type in ("hidden", "submit", "button"):
            continue

        if element.name == "textarea":
            field_type = FieldType.TEXTAREA
        elif element.name == "select":
            field_type = (
                FieldType.MULTISELECT if element.has_attr("multiple") else FieldType.SELECT
            )
        else:
            field_type = _TYPE_MAP.get(input_type, FieldType.TEXT)

        options: list[str] = []
        if field_type in (FieldType.SELECT, FieldType.MULTISELECT):
            options = [
                opt.get_text(strip=True)
                for opt in element.find_all("option")
                if opt.get("value")
            ]

        fields.append(
            ApplicationField(
                key=str(name),
                label=_label_for(form, element),
                field_type=field_type,
                required=_is_required(element),
                options=options,
            )
        )
    return fields


def build_multipart_data(
    *, field_values: dict[str, str], resume: ResumeAttachment, resume_field_name: str = "resume"
) -> tuple[dict[str, str], dict[str, tuple[str, bytes, str]]]:
    """Split mapped field values into httpx's `data=` / `files=` shape."""
    data = {key: value for key, value in field_values.items() if key != resume_field_name}
    files = {resume_field_name: (resume.filename, resume.content, resume.content_type)}
    return data, files


__all__ = ["find_form", "parse_form_fields", "build_multipart_data"]
