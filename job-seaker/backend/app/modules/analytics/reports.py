"""CSV / PDF report generation for the analytics dashboard.

Pure functions over plain dataclasses/schemas — mirrors
`app.modules.ats_resume_generator.documents`, so both formats are trivial to
unit test without a DB or S3.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from io import BytesIO, StringIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.modules.analytics.schemas import ExportApplicationRow

_CSV_HEADER = [
    "Company",
    "Job Title",
    "Status",
    "Source",
    "Location",
    "Applied At",
    "Updated At",
]


def generate_applications_csv(rows: list[ExportApplicationRow]) -> str:
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_CSV_HEADER)
    for row in rows:
        writer.writerow(
            [
                row.company_name,
                row.job_title,
                row.status.value,
                row.source,
                row.location or "",
                row.applied_at.isoformat(),
                row.updated_at.isoformat(),
            ]
        )
    return buffer.getvalue()


@dataclass
class AnalyticsReportContent:
    title: str
    generated_at: str
    date_range_label: str
    summary_rows: list[tuple[str, str]] = field(default_factory=list)
    top_companies: list[tuple[str, str, str, str]] = field(default_factory=list)
    top_skills: list[tuple[str, str]] = field(default_factory=list)
    top_countries: list[tuple[str, str]] = field(default_factory=list)


def generate_analytics_pdf(content: AnalyticsReportContent) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title=content.title,
    )
    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle(
        "AnalyticsHeading", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6
    )
    body_style = ParagraphStyle("AnalyticsBody", parent=styles["Normal"], spaceAfter=4)
    table_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
    )

    story: list = [
        Paragraph(content.title, styles["Title"]),
        Spacer(1, 4),
        Paragraph(f"Generated {content.generated_at} &middot; {content.date_range_label}", body_style),
        Spacer(1, 12),
    ]

    def add_table(heading: str, header_row: list[str], data_rows: list[tuple]) -> None:
        story.append(Paragraph(heading, heading_style))
        if not data_rows:
            story.append(Paragraph("No data for this period.", body_style))
            return
        table = Table([header_row, *[list(r) for r in data_rows]], hAlign="LEFT")
        table.setStyle(table_style)
        story.append(table)
        story.append(Spacer(1, 10))

    add_table("Summary", ["Metric", "Value"], content.summary_rows)
    add_table("Top Companies", ["Company", "Applications", "Interviews", "Offers"], content.top_companies)
    add_table("Top Skills", ["Skill", "Count"], content.top_skills)
    add_table("Top Countries", ["Country", "Applications"], content.top_countries)

    doc.build(story)
    return buffer.getvalue()


__all__ = ["generate_applications_csv", "AnalyticsReportContent", "generate_analytics_pdf"]
