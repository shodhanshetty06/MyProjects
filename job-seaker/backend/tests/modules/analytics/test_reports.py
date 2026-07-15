"""Unit tests for `app.modules.analytics.reports` — CSV/PDF generation,
no DB/HTTP I-O required.
"""
from __future__ import annotations

import csv
import io
import uuid
from datetime import UTC, datetime

from app.modules.analytics.reports import (
    AnalyticsReportContent,
    generate_analytics_pdf,
    generate_applications_csv,
)
from app.modules.analytics.schemas import ExportApplicationRow
from app.modules.tracking.models import ApplicationStatus

_ROW = ExportApplicationRow(
    id=uuid.uuid4(),
    company_name="Acme Corp",
    job_title="Backend Engineer",
    status=ApplicationStatus.INTERVIEW,
    source="manual",
    location="Berlin, Germany",
    applied_at=datetime(2026, 1, 5, tzinfo=UTC),
    updated_at=datetime(2026, 1, 10, tzinfo=UTC),
)


def test_generate_applications_csv_writes_header_and_row() -> None:
    csv_text = generate_applications_csv([_ROW])

    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)

    assert rows[0] == ["Company", "Job Title", "Status", "Source", "Location", "Applied At", "Updated At"]
    assert rows[1][0] == "Acme Corp"
    assert rows[1][2] == "interview"
    assert rows[1][4] == "Berlin, Germany"


def test_generate_applications_csv_empty_list_is_header_only() -> None:
    csv_text = generate_applications_csv([])
    rows = list(csv.reader(io.StringIO(csv_text)))
    assert len(rows) == 1


def test_generate_applications_csv_handles_missing_location() -> None:
    row = _ROW.model_copy(update={"location": None})
    csv_text = generate_applications_csv([row])
    rows = list(csv.reader(io.StringIO(csv_text)))
    assert rows[1][4] == ""


def test_generate_analytics_pdf_produces_a_valid_pdf() -> None:
    content = AnalyticsReportContent(
        title="Job Search Analytics Report",
        generated_at="2026-01-15 00:00 UTC",
        date_range_label="2025-10-15 to 2026-01-15",
        summary_rows=[("Total Applications", "12")],
        top_companies=[("Acme Corp", "3", "1", "1")],
        top_skills=[("Python", "2")],
        top_countries=[("Germany", "3")],
    )

    pdf_bytes = generate_analytics_pdf(content)

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 0


def test_generate_analytics_pdf_handles_empty_sections() -> None:
    content = AnalyticsReportContent(
        title="Job Search Analytics Report",
        generated_at="2026-01-15 00:00 UTC",
        date_range_label="2025-10-15 to 2026-01-15",
    )

    pdf_bytes = generate_analytics_pdf(content)

    assert pdf_bytes.startswith(b"%PDF")
