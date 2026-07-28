"""Celery application factory.

Run with:
    celery -A app.workers.celery_app worker --loglevel=info
    celery -A app.workers.celery_app beat --loglevel=info
"""

from __future__ import annotations

from celery import Celery
from celery.signals import task_postrun, worker_process_init

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
        "app.modules.ai_career_coach.tasks",
        "app.modules.workspace.tasks",
        "app.modules.daily_agent.tasks",
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
    # Long-running AI/embedding/Playwright tasks share queues with short
    # deterministic ones (scoring, bookkeeping). A high prefetch multiplier
    # lets one worker hoard several long tasks while siblings starve waiting
    # on the same queue, and early acking (the default) means a worker crash
    # or OOM mid-task (very plausible for Playwright-driven auto_apply, or
    # the embedding worker loading a sentence-transformers model) silently
    # drops the task instead of requeuing it. prefetch=1 + acks_late fixes
    # both: throughput here comes from worker concurrency, not prefetch.
    # task_reject_on_worker_lost pairs with acks_late so a task whose worker
    # is killed mid-execution (not just mid-wait) is requeued rather than
    # left stuck unacked forever.
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Bounds per-process memory growth from repeatedly loading ML models /
    # spawning Playwright browser instances across many tasks.
    worker_max_tasks_per_child=100,
    result_expires=3600,
    task_routes={
        "resume_parser.parse_resume": {"queue": "parsing"},
        "resume_parser.generate_embedding": {"queue": "embedding"},
        "resume_parser.calculate_score": {"queue": "embedding"},
        "resume_parser.run_ats_baseline_analysis": {"queue": "embedding"},
        "job_ingestion.sweep_companies": {"queue": "ingestion"},
        "job_ingestion.fetch_company_jobs": {"queue": "ingestion"},
        "job_ingestion.normalize_job": {"queue": "ingestion"},
        "job_ingestion.extract_job_skills": {"queue": "skills"},
        "job_ingestion.generate_job_embedding": {"queue": "embedding"},
        "matching.compute_resume_matches": {"queue": "matching"},
        "matching.refresh_recommendations": {"queue": "matching"},
        "matching.sweep_stale_matches": {"queue": "matching"},
        "matching.periodic_batch_sweep": {"queue": "matching"},
        "matching.match_new_job_for_active_users": {"queue": "matching"},
        "ats_resume_generator.optimize": {"queue": "ats_optimization"},
        "ats_resume_generator.generate_files": {"queue": "ats_files"},
        "auto_apply.submit_application": {"queue": "auto_apply"},
        "email_tracking.sweep_accounts": {"queue": "email_tracking"},
        "email_tracking.poll_account": {"queue": "email_tracking"},
        "email_tracking.fetch_and_classify_email": {"queue": "email_tracking"},
        "notification.send_email": {"queue": "notifications"},
        "notification.send_resume_ready": {"queue": "notifications"},
        "ai_career_coach.compute_dashboard_insights": {"queue": "matching"},
        "workspace.update_dashboard": {"queue": "matching"},
        "daily_agent.run_daily_scan": {"queue": "matching"},
        "daily_agent.run_for_user": {"queue": "matching"},
        "daily_agent.stage_pending_application": {"queue": "matching"},
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
    "daily-agent-run": {
        "task": "daily_agent.run_daily_scan",
        "schedule": settings.DAILY_AGENT_RUN_INTERVAL_MINUTES * 60,
    },
}


@worker_process_init.connect
def _register_event_handlers(**kwargs) -> None:  # noqa: ANN003 -- Celery signal signature
    """The `event_bus` (`app/events/bus.py`) is in-process — a worker
    process never runs FastAPI's lifespan (where this is otherwise called),
    so without this it would have zero subscribers and every
    `event_bus.publish()` a task makes (see `job_ingestion`/`matching`
    tasks) would silently no-op. Fires once per worker process (prefork
    spawns one call per child), matching `register_all_handlers()`'s
    once-per-process contract.
    """
    from app.events.handlers import register_all_handlers

    register_all_handlers()


@task_postrun.connect
def _reset_async_resources(**kwargs) -> None:  # noqa: ANN003 -- Celery signal signature
    """Every task body runs inside its own `asyncio.run(...)` call (see
    `app.modules.resume_parser.tasks` module docstring), which tears down its
    event loop when the task finishes. Both the SQLAlchemy async engine's
    pooled connections and the cached Redis client (`get_redis_client`) are
    bound to whatever loop was running when they were opened — a loop that
    is already closed by the time this signal fires. There is no way to
    gracefully close a connection whose loop is gone (attempting to raises
    "Future attached to a different loop" / "Event loop is closed"), so
    instead we abandon the stale connections outright: `dispose(close=False)`
    drops the pool's references without touching the dead connections, and
    clearing the Redis client cache forces a brand new client to be created
    lazily on the next task's own loop.
    """
    from app.db.session import engine
    from app.integrations.redis_client import get_redis_client

    engine.sync_engine.dispose(close=False)
    get_redis_client.cache_clear()
