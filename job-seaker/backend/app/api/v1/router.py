"""Aggregates every v1 module router into a single APIRouter.

Each feature module owns its own `router.py`; this file only imports and
mounts them under a consistent prefix/tag. Adding a new module means
adding one import + one `include_router` line here.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.storage import router as storage_router
from app.modules.admin.router import router as admin_router
from app.modules.ai_career_coach.router import router as ai_career_coach_router
from app.modules.analytics.router import router as analytics_router
from app.modules.ats_resume_generator.router import router as ats_resume_generator_router
from app.modules.auth.router import router as auth_router
from app.modules.auto_apply.router import router as auto_apply_router
from app.modules.email_tracking.router import router as email_tracking_router
from app.modules.job_ingestion.router import router as job_ingestion_router
from app.modules.matching.router import router as matching_router
from app.modules.notification.router import router as notification_router
from app.modules.resume_parser.router import router as resume_parser_router
from app.modules.search.router import router as search_router
from app.modules.tracking.router import router as tracking_router
from app.modules.user_profile.router import onboarding_router, router as user_profile_router
from app.modules.workspace.router import router as workspace_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(storage_router)
api_router.include_router(auth_router)
api_router.include_router(user_profile_router)
api_router.include_router(onboarding_router)
api_router.include_router(resume_parser_router)
api_router.include_router(job_ingestion_router)
api_router.include_router(search_router)
api_router.include_router(matching_router)
api_router.include_router(ats_resume_generator_router)
api_router.include_router(auto_apply_router)
api_router.include_router(tracking_router)
api_router.include_router(email_tracking_router)
api_router.include_router(notification_router)
api_router.include_router(ai_career_coach_router)
api_router.include_router(analytics_router)
api_router.include_router(workspace_router)
api_router.include_router(admin_router)
