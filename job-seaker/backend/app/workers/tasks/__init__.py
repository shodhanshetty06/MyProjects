"""Task registry package.

Module-specific Celery tasks live under `app/modules/<name>/tasks.py` and
are wired into `celery_app.include` as they're implemented. This package
is the default include target and stays empty in Phase 0.
"""
