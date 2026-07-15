"""Plain-text extraction from uploaded resume files.

No framework dependency — these are pure functions over raw bytes so they're
trivial to unit test and reusable from both the API layer (upload
validation) and the Celery parsing task.
"""
from __future__ import annotations

import io

import pypdf
from docx import Document

from app.core.exceptions import ValidationError
from app.modules.resume_parser.models import ResumeContentType


def extract_text_from_pdf(data: bytes) -> str:
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pypdf raises various exception types on malformed files
        raise ValidationError(f"Could not read PDF file: {exc}") from exc
    return "\n".join(pages).strip()


def extract_text_from_docx(data: bytes) -> str:
    try:
        document = Document(io.BytesIO(data))
        paragraphs = [p.text for p in document.paragraphs]
    except Exception as exc:  # python-docx raises various exception types on malformed files
        raise ValidationError(f"Could not read DOCX file: {exc}") from exc
    return "\n".join(paragraphs).strip()


def extract_text(content_type: ResumeContentType, data: bytes) -> str:
    if content_type == ResumeContentType.PDF:
        text = extract_text_from_pdf(data)
    elif content_type == ResumeContentType.DOCX:
        text = extract_text_from_docx(data)
    else:  # pragma: no cover — guarded by the enum, kept as a defensive fallback
        raise ValidationError(f"Unsupported resume content type: {content_type}")

    if not text:
        raise ValidationError("No extractable text found in the uploaded resume.")
    return text
