"""Resume Parser module — API routes (v1)."""
from __future__ import annotations

import math
import uuid

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import CurrentUserIdDep, DBSessionDep, StorageProviderDep
from app.modules.resume_parser.models import Resume, ResumeVersion
from app.modules.resume_parser.repository import (
    ResumeParsedDataRepository,
    ResumeRepository,
    ResumeVersionRepository,
)
from app.modules.resume_parser.schemas import (
    ParsedResumeData,
    ResumeRead,
    ResumeUpdateRequest,
    ResumeVersionDetail,
    ResumeVersionSummary,
)
from app.modules.resume_parser.service import ResumeParserService
from app.schemas.common import BaseResponse, PaginatedResponse, PaginationMeta, PaginationParams

router = APIRouter(prefix="/resumes", tags=["Resume Parser"])


def get_resume_parser_service(
    session: DBSessionDep, storage_provider: StorageProviderDep
) -> ResumeParserService:
    return ResumeParserService(
        resume_repository=ResumeRepository(session),
        version_repository=ResumeVersionRepository(session),
        storage_provider=storage_provider,
        parsed_data_repository=ResumeParsedDataRepository(session),
    )


ServiceDep = Depends(get_resume_parser_service)


def _version_summary(version: ResumeVersion) -> ResumeVersionSummary:
    return ResumeVersionSummary.model_validate(version)


def _resume_read(resume: Resume, current_version: ResumeVersion | None = None) -> ResumeRead:
    current = current_version or next((v for v in resume.versions if v.is_current), None)
    return ResumeRead(
        id=resume.id,
        user_id=resume.user_id,
        title=resume.title,
        created_at=resume.created_at,
        updated_at=resume.updated_at,
        current_version=_version_summary(current) if current else None,
    )


def _version_detail(version: ResumeVersion) -> ResumeVersionDetail:
    parsed = (
        ParsedResumeData(
            contact=version.parsed_data.contact,
            summary=version.parsed_data.summary,
            skills=version.parsed_data.skills,
            experience=version.parsed_data.experience,
            education=version.parsed_data.education,
            projects=version.parsed_data.projects,
            certifications=version.parsed_data.certifications,
            achievements=version.parsed_data.achievements,
            languages=version.parsed_data.languages,
        )
        if version.parsed_data
        else None
    )
    return ResumeVersionDetail(
        **ResumeVersionSummary.model_validate(version).model_dump(),
        parsed_data=parsed,
        has_embedding=version.embedding is not None,
    )


@router.post("", response_model=BaseResponse[ResumeRead], status_code=201)
async def upload_resume(
    user_id: CurrentUserIdDep,
    service: ResumeParserService = ServiceDep,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
) -> BaseResponse[ResumeRead]:
    data = await file.read()
    resume, version = await service.upload_resume(
        user_id=user_id,
        filename=file.filename or "resume",
        content_type_header=file.content_type,
        data=data,
        title=title,
    )
    return BaseResponse(
        message="Resume uploaded and queued for parsing.",
        data=_resume_read(resume, current_version=version),
    )


@router.get("", response_model=PaginatedResponse[ResumeRead])
async def list_resumes(
    user_id: CurrentUserIdDep,
    pagination: PaginationParams = Depends(),
    service: ResumeParserService = ServiceDep,
) -> PaginatedResponse[ResumeRead]:
    resumes, total = await service.list_resumes(
        user_id=user_id, offset=pagination.offset, limit=pagination.page_size
    )
    return PaginatedResponse(
        data=[_resume_read(r) for r in resumes],
        pagination=PaginationMeta(
            page=pagination.page,
            page_size=pagination.page_size,
            total_items=total,
            total_pages=math.ceil(total / pagination.page_size) if total else 0,
        ),
    )


@router.get("/{resume_id}", response_model=BaseResponse[ResumeRead])
async def get_resume(
    resume_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: ResumeParserService = ServiceDep,
) -> BaseResponse[ResumeRead]:
    resume = await service.get_resume(user_id=user_id, resume_id=resume_id)
    return BaseResponse(data=_resume_read(resume))


@router.patch("/{resume_id}", response_model=BaseResponse[ResumeRead])
async def update_resume(
    resume_id: uuid.UUID,
    payload: ResumeUpdateRequest,
    user_id: CurrentUserIdDep,
    service: ResumeParserService = ServiceDep,
) -> BaseResponse[ResumeRead]:
    resume = await service.update_resume(user_id=user_id, resume_id=resume_id, title=payload.title)
    return BaseResponse(message="Resume updated.", data=_resume_read(resume))


@router.delete("/{resume_id}", response_model=BaseResponse[None])
async def delete_resume(
    resume_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: ResumeParserService = ServiceDep,
) -> BaseResponse[None]:
    await service.delete_resume(user_id=user_id, resume_id=resume_id)
    return BaseResponse(message="Resume deleted.")


@router.post("/{resume_id}/versions", response_model=BaseResponse[ResumeVersionSummary], status_code=201)
async def upload_resume_version(
    resume_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: ResumeParserService = ServiceDep,
    file: UploadFile = File(...),
) -> BaseResponse[ResumeVersionSummary]:
    data = await file.read()
    version = await service.upload_new_version(
        user_id=user_id,
        resume_id=resume_id,
        filename=file.filename or "resume",
        content_type_header=file.content_type,
        data=data,
    )
    return BaseResponse(message="New resume version uploaded and queued for parsing.", data=_version_summary(version))


@router.get("/{resume_id}/versions", response_model=BaseResponse[list[ResumeVersionSummary]])
async def list_resume_versions(
    resume_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: ResumeParserService = ServiceDep,
) -> BaseResponse[list[ResumeVersionSummary]]:
    versions = await service.list_versions(user_id=user_id, resume_id=resume_id)
    return BaseResponse(data=[_version_summary(v) for v in versions])


@router.get("/{resume_id}/versions/{version_id}", response_model=BaseResponse[ResumeVersionDetail])
async def get_resume_version(
    resume_id: uuid.UUID,
    version_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: ResumeParserService = ServiceDep,
) -> BaseResponse[ResumeVersionDetail]:
    version = await service.get_version(
        user_id=user_id, resume_id=resume_id, version_id=version_id, with_details=True
    )
    return BaseResponse(data=_version_detail(version))


@router.patch(
    "/{resume_id}/versions/{version_id}/parsed-data", response_model=BaseResponse[ResumeVersionDetail]
)
async def update_parsed_data(
    resume_id: uuid.UUID,
    version_id: uuid.UUID,
    payload: ParsedResumeData,
    user_id: CurrentUserIdDep,
    service: ResumeParserService = ServiceDep,
) -> BaseResponse[ResumeVersionDetail]:
    """Saves the Profile Review screen's edits over the AI-extracted resume
    data — accepted values pass through unchanged, edited/deleted/added
    entries land here as the corrected `ParsedResumeData` shape.
    """
    version = await service.update_parsed_data(
        user_id=user_id, resume_id=resume_id, version_id=version_id, data=payload
    )
    return BaseResponse(message="Profile updated.", data=_version_detail(version))


@router.put(
    "/{resume_id}/versions/{version_id}/activate", response_model=BaseResponse[ResumeVersionSummary]
)
async def activate_resume_version(
    resume_id: uuid.UUID,
    version_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: ResumeParserService = ServiceDep,
) -> BaseResponse[ResumeVersionSummary]:
    version = await service.activate_version(user_id=user_id, resume_id=resume_id, version_id=version_id)
    return BaseResponse(message="Resume version activated.", data=_version_summary(version))


@router.delete("/{resume_id}/versions/{version_id}", response_model=BaseResponse[None])
async def delete_resume_version(
    resume_id: uuid.UUID,
    version_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: ResumeParserService = ServiceDep,
) -> BaseResponse[None]:
    await service.delete_version(user_id=user_id, resume_id=resume_id, version_id=version_id)
    return BaseResponse(message="Resume version deleted.")


@router.get("/{resume_id}/versions/{version_id}/download", response_model=BaseResponse[dict])
async def download_resume_version(
    resume_id: uuid.UUID,
    version_id: uuid.UUID,
    user_id: CurrentUserIdDep,
    service: ResumeParserService = ServiceDep,
) -> BaseResponse[dict]:
    url = await service.get_download_url(user_id=user_id, resume_id=resume_id, version_id=version_id)
    return BaseResponse(data={"download_url": url})
