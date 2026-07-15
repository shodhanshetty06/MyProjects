"""ATS Resume Generator module — API routes (v1)."""
from __future__ import annotations

import math
import uuid

from fastapi import APIRouter, Depends, Query

from app.api.deps import CurrentUserIdDep, DBSessionDep, S3StorageDep
from app.modules.ats_resume_generator.models import ATSFileFormat, ATSResume, ATSResumeVersion
from app.modules.ats_resume_generator.repository import (
    ATSResumeRepository,
    ATSResumeVersionRepository,
)
from app.modules.ats_resume_generator.schemas import (
    ATSResumeCreateRequest,
    ATSResumeDownloadResponse,
    ATSResumeRead,
    ATSResumeUpdateRequest,
    ATSResumeVersionCreateRequest,
    ATSResumeVersionDetail,
    ATSResumeVersionDiff,
    ATSResumeVersionSummary,
)
from app.modules.ats_resume_generator.service import ATSResumeGeneratorService
from app.schemas.common import BaseResponse, PaginatedResponse, PaginationMeta, PaginationParams

router = APIRouter(prefix="/ats-resumes", tags=["ATS Resume Generator"])


def get_ats_resume_generator_service(
    session: DBSessionDep, s3_storage: S3StorageDep
) -> ATSResumeGeneratorService:
    return ATSResumeGeneratorService(
        ats_resume_repository=ATSResumeRepository(session),
        version_repository=ATSResumeVersionRepository(session),
        s3_storage=s3_storage,
    )


ServiceDep = Depends(get_ats_resume_generator_service)


def _optional_float(value: object) -> float | None:
    return float(value) if value is not None else None  # type: ignore[arg-type]


def _version_summary(version: ATSResumeVersion) -> ATSResumeVersionSummary:
    return ATSResumeVersionSummary(
        id=version.id,
        version_number=version.version_number,
        is_current=version.is_current,
        status=version.status,
        keyword_score=_optional_float(version.keyword_score),
        ats_score=_optional_float(version.ats_score),
        error_message=version.error_message,
        available_formats=[f.format for f in version.files],
        created_at=version.created_at,
    )


def _version_detail(version: ATSResumeVersion) -> ATSResumeVersionDetail:
    return ATSResumeVersionDetail(
        **_version_summary(version).model_dump(),
        job_description=version.job_description,
        source_resume_version_id=version.source_resume_version_id,
        original_summary=version.original_summary,
        optimized_summary=version.optimized_summary,
        original_skills=version.original_skills,
        optimized_skills=version.optimized_skills,
        original_projects=version.original_projects,
        optimized_projects=version.optimized_projects,
        experience=version.experience,
        education=version.education,
        certifications=version.certifications,
        languages=version.languages,
        achievements=version.achievements,
        keywords=version.keywords,
        matched_keywords=version.matched_keywords,
        missing_keywords=version.missing_keywords,
        score_breakdown=version.score_breakdown,
        model_used=version.model_used,
        updated_at=version.updated_at,
    )


def _ats_resume_read(
    ats_resume: ATSResume, current_version: ATSResumeVersion | None = None
) -> ATSResumeRead:
    current = current_version or next((v for v in ats_resume.versions if v.is_current), None)
    return ATSResumeRead(
        id=ats_resume.id,
        user_id=ats_resume.user_id,
        source_resume_id=ats_resume.source_resume_id,
        target_job_id=ats_resume.target_job_id,
        title=ats_resume.title,
        job_title=ats_resume.job_title,
        company_name=ats_resume.company_name,
        created_at=ats_resume.created_at,
        updated_at=ats_resume.updated_at,
        current_version=_version_summary(current) if current else None,
    )


@router.post("", response_model=BaseResponse[ATSResumeRead], status_code=201)
async def create_ats_resume(
    payload: ATSResumeCreateRequest,
    user_id: CurrentUserIdDep,
    service: ATSResumeGeneratorService = ServiceDep,
) -> BaseResponse[ATSResumeRead]:
    ats_resume, version = await service.create_ats_resume(user_id=user_id, payload=payload)
    return BaseResponse(
        message="ATS resume created and queued for optimization.",
        data=_ats_resume_read(ats_resume, current_version=version),
    )


@router.get("", response_model=PaginatedResponse[ATSResumeRead])
async def list_ats_resumes(
    user_id: CurrentUserIdDep,
    pagination: PaginationParams = Depends(),
    service: ATSResumeGeneratorService = ServiceDep,
) -> PaginatedResponse[ATSResumeRead]:
    resumes, total = await service.list_ats_resumes(
        user_id=user_id, offset=pagination.offset, limit=pagination.page_size
    )
    return PaginatedResponse(
        data=[_ats_resume_read(r) for r in resumes],
        pagination=PaginationMeta(
            page=pagination.page,
            page_size=pagination.page_size,
            total_items=total,
            total_pages=math.ceil(total / pagination.page_size) if total else 0,
        ),
    )


@router.get("/{ats_resume_id}", response_model=BaseResponse[ATSResumeRead])
async def get_ats_resume(
    ats_resume_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: ATSResumeGeneratorService = ServiceDep,
) -> BaseResponse[ATSResumeRead]:
    ats_resume = await service.get_ats_resume(user_id=user_id, ats_resume_id=ats_resume_id)
    return BaseResponse(data=_ats_resume_read(ats_resume))


@router.patch("/{ats_resume_id}", response_model=BaseResponse[ATSResumeRead])
async def update_ats_resume(
    ats_resume_id: uuid.UUID,
    payload: ATSResumeUpdateRequest,
    user_id: CurrentUserIdDep,
    service: ATSResumeGeneratorService = ServiceDep,
) -> BaseResponse[ATSResumeRead]:
    ats_resume = await service.update_ats_resume(
        user_id=user_id, ats_resume_id=ats_resume_id, title=payload.title
    )
    return BaseResponse(message="ATS resume updated.", data=_ats_resume_read(ats_resume))


@router.delete("/{ats_resume_id}", response_model=BaseResponse[None])
async def delete_ats_resume(
    ats_resume_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: ATSResumeGeneratorService = ServiceDep,
) -> BaseResponse[None]:
    await service.delete_ats_resume(user_id=user_id, ats_resume_id=ats_resume_id)
    return BaseResponse(message="ATS resume deleted.")


@router.post(
    "/{ats_resume_id}/versions",
    response_model=BaseResponse[ATSResumeVersionSummary],
    status_code=201,
)
async def create_ats_resume_version(
    ats_resume_id: uuid.UUID,
    payload: ATSResumeVersionCreateRequest,
    user_id: CurrentUserIdDep,
    service: ATSResumeGeneratorService = ServiceDep,
) -> BaseResponse[ATSResumeVersionSummary]:
    version = await service.create_new_version(
        user_id=user_id, ats_resume_id=ats_resume_id, payload=payload
    )
    return BaseResponse(
        message="New ATS resume version queued for optimization.", data=_version_summary(version)
    )


@router.get("/{ats_resume_id}/versions", response_model=BaseResponse[list[ATSResumeVersionSummary]])
async def list_ats_resume_versions(
    ats_resume_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: ATSResumeGeneratorService = ServiceDep,
) -> BaseResponse[list[ATSResumeVersionSummary]]:
    versions = await service.list_versions(user_id=user_id, ats_resume_id=ats_resume_id)
    return BaseResponse(data=[_version_summary(v) for v in versions])


@router.get(
    "/{ats_resume_id}/versions/{version_id}", response_model=BaseResponse[ATSResumeVersionDetail]
)
async def get_ats_resume_version(
    ats_resume_id: uuid.UUID,
    version_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: ATSResumeGeneratorService = ServiceDep,
) -> BaseResponse[ATSResumeVersionDetail]:
    version = await service.get_version(
        user_id=user_id, ats_resume_id=ats_resume_id, version_id=version_id
    )
    return BaseResponse(data=_version_detail(version))


@router.put(
    "/{ats_resume_id}/versions/{version_id}/activate",
    response_model=BaseResponse[ATSResumeVersionSummary],
)
async def activate_ats_resume_version(
    ats_resume_id: uuid.UUID,
    version_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: ATSResumeGeneratorService = ServiceDep,
) -> BaseResponse[ATSResumeVersionSummary]:
    version = await service.activate_version(
        user_id=user_id, ats_resume_id=ats_resume_id, version_id=version_id
    )
    return BaseResponse(message="ATS resume version activated.", data=_version_summary(version))


@router.delete("/{ats_resume_id}/versions/{version_id}", response_model=BaseResponse[None])
async def delete_ats_resume_version(
    ats_resume_id: uuid.UUID,
    version_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: ATSResumeGeneratorService = ServiceDep,
) -> BaseResponse[None]:
    await service.delete_version(user_id=user_id, ats_resume_id=ats_resume_id, version_id=version_id)
    return BaseResponse(message="ATS resume version deleted.")


@router.get(
    "/{ats_resume_id}/versions/{version_id}/diff", response_model=BaseResponse[ATSResumeVersionDiff]
)
async def get_ats_resume_version_diff(
    ats_resume_id: uuid.UUID,
    version_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: ATSResumeGeneratorService = ServiceDep,
    compare_to: str = Query(default="source", description="'source' or another version's id"),
) -> BaseResponse[ATSResumeVersionDiff]:
    diff = await service.get_diff(
        user_id=user_id, ats_resume_id=ats_resume_id, version_id=version_id, compare_to=compare_to
    )
    return BaseResponse(data=diff)


@router.get(
    "/{ats_resume_id}/versions/{version_id}/download/{format}",
    response_model=BaseResponse[ATSResumeDownloadResponse],
)
async def download_ats_resume_version(
    ats_resume_id: uuid.UUID,
    version_id: uuid.UUID,
    format: ATSFileFormat,
    user_id: CurrentUserIdDep,
    service: ATSResumeGeneratorService = ServiceDep,
) -> BaseResponse[ATSResumeDownloadResponse]:
    url = await service.get_download_url(
        user_id=user_id, ats_resume_id=ats_resume_id, version_id=version_id, format=format
    )
    return BaseResponse(data=ATSResumeDownloadResponse(format=format, download_url=url))
