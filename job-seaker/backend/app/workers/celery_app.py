"""Celery application factory.

Run with:
    celery -A app.workers.celery_app worker --loglevel=info
    celery -A app.workers.celery_app beat --loglevel=info
"""

from __future__ import annotations

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "job_platform",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        # Task modules are registered here as modules add real tasks.
        "app.modules.resume_parser.tasks",
        "app.modules.job_ingestion.tasks",
        "app.modules.matching.tasks",
        "app.modules.ats_resume_generator.tasks",
        "app.modules.auto_apply.tasks",
        "app.modules.email_tracking.tasks",
        "app.modules.notification.tasks",
        "app.workers.tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,
    task_soft_time_limit=25 * 60,
    worker_prefetch_multiplier=4,
    result_expires=3600,
    task_routes={
        "resume_parser.parse_resume": {"queue": "parsing"},
        "resume_parser.generate_embedding": {"queue": "embedding"},
        "job_ingestion.sweep_companies": {"queue": "ingestion"},
        "job_ingestion.fetch_company_jobs": {"queue": "ingestion"},
        "job_ingestion.normalize_job": {"queue": "ingestion"},
        "job_ingestion.extract_job_skills": {"queue": "skills"},
        "job_ingestion.generate_job_embedding": {"queue": "embedding"},
        "matching.compute_resume_matches": {"queue": "matching"},
        "matching.refresh_recommendations": {"queue": "matching"},
        "matching.sweep_stale_matches": {"queue": "matching"},
        "matching.periodic_batch_sweep": {"queue": "matching"},
        "ats_resume_generator.optimize": {"queue": "ats_optimization"},
        "ats_resume_generator.generate_files": {"queue": "ats_files"},
        "auto_apply.submit_application": {"queue": "auto_apply"},
        "email_tracking.sweep_accounts": {"queue": "email_tracking"},
        "email_tracking.poll_account": {"queue": "email_tracking"},
        "email_tracking.fetch_and_classify_email": {"queue": "email_tracking"},
        "notification.send_email": {"queue": "notifications"},
    },
)

# Periodic task schedule.
# - job-ingestion-sweep: nightly (by default) crawl sweep, interval
#   configurable via JOB_INGESTION_SWEEP_INTERVAL_MINUTES.
# - matching-periodic-batch-sweep: recompute matches (and, via task chaining,
#   recommendations) for every user with an active preference + current
#   resume, on MATCHING_RECOMMENDATION_REFRESH_INTERVAL_MINUTES.
# - matching-stale-sweep: hourly bookkeeping pass marking matches older than
#   MATCHING_STALE_MATCH_HOURS as stale (cheap UPDATE, no recompute).
# - gmail-sync-sweep: polls every connected Gmail account for new mail,
#   interval configurable via GMAIL_SYNC_INTERVAL_MINUTES.
celery_app.conf.beat_schedule = {
    "job-ingestion-sweep": {
        "task": "job_ingestion.sweep_companies",
        "schedule": settings.JOB_INGESTION_SWEEP_INTERVAL_MINUTES * 60,
    },
    "matching-periodic-batch-sweep": {
        "task": "matching.periodic_batch_sweep",
        "schedule": settings.MATCHING_RECOMMENDATION_REFRESH_INTERVAL_MINUTES * 60,
    },
    "matching-stale-sweep": {
        "task": "matching.sweep_stale_matches",
        "schedule": 3600,
    },
    "gmail-sync-sweep": {
        "task": "email_tracking.sweep_accounts",
        "schedule": settings.GMAIL_SYNC_INTERVAL_MINUTES * 60,
    },
}
