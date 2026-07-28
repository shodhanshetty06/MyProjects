
# Pinned to bookworm rather than the floating `slim` tag (now Debian
# trixie) — Playwright's `--with-deps` apt package list below 404s on
# trixie (e.g. `ttf-unifont`/`ttf-ubuntu-font-family` were renamed/dropped).
FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    libreoffice-writer \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Playwright's own browser binary + its OS-level dependencies (Workday and
# the generic fallback adapter in `auto_apply` drive headless Chromium).
RUN playwright install --with-deps chromium

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
