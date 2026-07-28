"""ATS Resume Generator module — DOCX -> PDF conversion.

Per the "generate PDF from the DOCX, not a second independent pass" design
goal, this shells out to LibreOffice headless (`soffice --headless
--convert-to pdf`) to convert the already-rendered DOCX bytes, so the two
formats can never visually drift and the PDF is never built by placing raw
text/paragraphs on a canvas. `soffice` is installed in the backend Docker
image (see `backend/Dockerfile`) — the same image `worker-ats-resume` in
docker-compose.yml runs, so this works in the Celery worker that actually
calls it.

Synchronous/blocking by design — callers on the async task path should wrap
this in `asyncio.to_thread(...)` so the LibreOffice subprocess (typically
1-3s) doesn't block the event loop.
"""
from __future__ import annotations

import subprocess
import tempfile
import uuid
from pathlib import Path

_CONVERT_TIMEOUT_SECONDS = 60


def convert_docx_to_pdf(docx_bytes: bytes) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        docx_path = tmp_path / f"{uuid.uuid4().hex}.docx"
        docx_path.write_bytes(docx_bytes)

        result = subprocess.run(
            [
                "soffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(tmp_path),
                str(docx_path),
            ],
            capture_output=True,
            timeout=_CONVERT_TIMEOUT_SECONDS,
        )

        pdf_path = docx_path.with_suffix(".pdf")
        if result.returncode != 0 or not pdf_path.exists():
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"LibreOffice DOCX->PDF conversion failed: {stderr or 'no output produced'}")

        return pdf_path.read_bytes()


__all__ = ["convert_docx_to_pdf"]
