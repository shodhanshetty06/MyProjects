"""ATS Resume Generator module — visual template registry.

Pure data, no rendering logic (that's `docx_renderer.py`) — so the frontend
can fetch `list_templates()` metadata (via the router) to build a picker/
preview UI without ever generating a real file. Every template is a strict,
single-column, ATS-safe layout: no tables, text boxes, columns, or images.
`accent_color` is only ever used for a thin rule under the name/section
headings, never as a fill or background, per the "no unnecessary
decorations" requirement.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemplateSpec:
    id: str
    name: str
    description: str
    body_font: str
    heading_font: str
    base_font_size_pt: int
    heading_color: str  # hex, e.g. "#1F3864"
    accent_color: str | None  # hex, used only for a thin rule; None = no rule
    margins_inches: float
    heading_style: str  # "uppercase-rule" | "small-caps" | "bold-line" | "bold-plain"
    bullet_char: str
    section_spacing_pt: int
    line_spacing: float


TEMPLATES: dict[str, TemplateSpec] = {
    "modern_ats": TemplateSpec(
        id="modern_ats",
        name="Modern ATS",
        description="Clean sans-serif layout with a subtle navy rule under the name — the safest, most broadly ATS-compatible default.",
        body_font="Calibri",
        heading_font="Calibri",
        base_font_size_pt=11,
        heading_color="#1F3864",
        accent_color="#1F3864",
        margins_inches=0.7,
        heading_style="uppercase-rule",
        bullet_char="•",
        section_spacing_pt=10,
        line_spacing=1.08,
    ),
    "executive": TemplateSpec(
        id="executive",
        name="Executive",
        description="Serif typography and generous spacing for a senior/leadership presence.",
        body_font="Georgia",
        heading_font="Georgia",
        base_font_size_pt=11,
        heading_color="#33302A",
        accent_color="#8A7355",
        margins_inches=0.8,
        heading_style="bold-line",
        bullet_char="•",
        section_spacing_pt=14,
        line_spacing=1.15,
    ),
    "minimal": TemplateSpec(
        id="minimal",
        name="Minimal",
        description="Black-on-white, no accent color, tightest spacing — pure content density.",
        body_font="Arial",
        heading_font="Arial",
        base_font_size_pt=10.5,
        heading_color="#000000",
        accent_color=None,
        margins_inches=0.6,
        heading_style="bold-plain",
        bullet_char="-",
        section_spacing_pt=6,
        line_spacing=1.0,
    ),
    "professional": TemplateSpec(
        id="professional",
        name="Professional",
        description="Classic corporate look with small-caps section headings in deep blue-gray.",
        body_font="Calibri",
        heading_font="Calibri",
        base_font_size_pt=11,
        heading_color="#2E4057",
        accent_color="#2E4057",
        margins_inches=0.75,
        heading_style="small-caps",
        bullet_char="•",
        section_spacing_pt=10,
        line_spacing=1.1,
    ),
    "tech": TemplateSpec(
        id="tech",
        name="Tech",
        description="Sans-serif with a single teal accent rule — built for SWE/data roles.",
        body_font="Helvetica",
        heading_font="Helvetica",
        base_font_size_pt=10.5,
        heading_color="#0F4C4C",
        accent_color="#0E8C8C",
        margins_inches=0.65,
        heading_style="uppercase-rule",
        bullet_char="‣",
        section_spacing_pt=8,
        line_spacing=1.05,
    ),
    "consulting": TemplateSpec(
        id="consulting",
        name="Consulting",
        description="Dense, case-style layout with right-aligned dates and a burgundy accent — McKinsey/BCG resume conventions.",
        body_font="Garamond",
        heading_font="Garamond",
        base_font_size_pt=11,
        heading_color="#5B1A1A",
        accent_color="#8C2E2E",
        margins_inches=0.7,
        heading_style="bold-line",
        bullet_char="•",
        section_spacing_pt=8,
        line_spacing=1.05,
    ),
}

DEFAULT_TEMPLATE_ID = "modern_ats"


def get_template(template_id: str | None) -> TemplateSpec:
    return TEMPLATES.get(template_id or DEFAULT_TEMPLATE_ID, TEMPLATES[DEFAULT_TEMPLATE_ID])


def list_templates() -> list[TemplateSpec]:
    return list(TEMPLATES.values())


__all__ = ["TemplateSpec", "TEMPLATES", "DEFAULT_TEMPLATE_ID", "get_template", "list_templates"]
