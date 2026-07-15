"""Unit tests for plain-text extraction from PDF/DOCX resume files.

No DB, network, or Docker services required — files are generated in memory.
"""
from __future__ import annotations

import io

import pytest
from docx import Document

from app.core.exceptions import ValidationError
from app.modules.resume_parser.extraction import (
    extract_text,
    extract_text_from_docx,
    extract_text_from_pdf,
)
from app.modules.resume_parser.models import ResumeContentType


def _make_minimal_pdf(text: str) -> bytes:
    """Hand-build a minimal, byte-accurate single-page PDF containing `text`.

    pypdf can be lenient about malformed xref tables, but computing real
    offsets keeps this test honest rather than relying on pypdf's recovery
    path.
    """

    def obj(num: int, body: str) -> bytes:
        return f"{num} 0 obj\n{body}\nendobj\n".encode()

    content = f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET"
    stream_obj = f"5 0 obj\n<< /Length {len(content)} >>\nstream\n{content}\nendstream\nendobj\n".encode()

    header = b"%PDF-1.4\n"
    objects = [
        obj(1, "<< /Type /Catalog /Pages 2 0 R >>"),
        obj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        obj(
            3,
            "<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
            "/MediaBox [0 0 200 200] /Contents 5 0 R >>",
        ),
        obj(4, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
        stream_obj,
    ]

    body = header
    offsets = [0]
    for o in objects:
        offsets.append(len(body))
        body += o

    xref_offset = len(body)
    xref = "xref\n0 6\n0000000000 65535 f \n"
    for offset in offsets[1:]:
        xref += f"{offset:010d} 00000 n \n"
    trailer = f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF"

    return body + xref.encode() + trailer.encode()


def _make_docx(paragraphs: list[str]) -> bytes:
    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_extract_text_from_pdf_returns_page_text() -> None:
    pdf_bytes = _make_minimal_pdf("Jane Doe Software Engineer")
    assert extract_text_from_pdf(pdf_bytes) == "Jane Doe Software Engineer"


def test_extract_text_from_pdf_raises_on_garbage_bytes() -> None:
    with pytest.raises(ValidationError):
        extract_text_from_pdf(b"not a real pdf")


def test_extract_text_from_docx_joins_paragraphs() -> None:
    docx_bytes = _make_docx(["Jane Doe", "Software Engineer", "jane@example.com"])
    text = extract_text_from_docx(docx_bytes)
    assert "Jane Doe" in text
    assert "Software Engineer" in text
    assert "jane@example.com" in text


def test_extract_text_from_docx_raises_on_garbage_bytes() -> None:
    with pytest.raises(ValidationError):
        extract_text_from_docx(b"not a real docx")


def test_extract_text_dispatches_on_content_type() -> None:
    docx_bytes = _make_docx(["Resume content"])
    assert extract_text(ResumeContentType.DOCX, docx_bytes) == "Resume content"

    pdf_bytes = _make_minimal_pdf("Resume content")
    assert extract_text(ResumeContentType.PDF, pdf_bytes) == "Resume content"


def test_extract_text_raises_on_empty_result() -> None:
    empty_docx = _make_docx([])
    with pytest.raises(ValidationError):
        extract_text(ResumeContentType.DOCX, empty_docx)
