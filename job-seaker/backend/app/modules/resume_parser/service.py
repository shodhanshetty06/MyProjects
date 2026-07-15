"""Resume Parser module — application/use-case layer."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Sequence

from app.core.config import get_settings
from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.core.logging import get_logger
from app.integrations.s3_client import S3Storage
from app.modules.resume_parser.models import (
    Resume,
    ResumeContentType,
    ResumeVersion,
    ResumeVersionStatus,
)
from app.modules.resume_parser.repository import ResumeRepository, ResumeVersionRepository

logger = get_logger(__name__)

_CONTENT_TYPE_MAP: dict[str, ResumeContentType] = {
    "application/pdf": ResumeContentType.PDF,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ResumeContentType.DOCX,
}
_EXTENSION_MAP: dict[str, ResumeContentType] = {
    ".pdf": ResumeContentType.PDF,
    ".docx": ResumeContentType.DOCX,
}


def _resolve_content_type(filename: str, content_type_header: str | None) -> ResumeContentType:
    if content_type_header and content_type_header in _CONTENT_TYPE_MAP:
        return _CONTENT_TYPE_MAP[content_type_header]

    suffix = filename[filename.rfind(".") :].lower() if "." in filename else ""
    if suffix in _EXTENSION_MAP:
        return _EXTENSION_MAP[suffix]

    raise ValidationError(
        "Unsupported file type. Only PDF and DOCX resumes are accepted.",
        error_code="unsupported_file_type",
    )


def _s3_key(user_id: uuid.UUID, resume_id: uuid.UUID, version_id: uuid.UUID, filename: str) -> str:
    return f"resumes/{user_id}/{resume_id}/{version_id}/{filename}"


class ResumeParserService:
    """Business logic orchestration for the Resume Parser module."""

    def __init__(
        self,
        resume_repository: ResumeRepository,
        version_repository: ResumeVersionRepository,
        s3_storage: S3Storage,
    ) -> None:
        self.resume_repository = resume_repository
        self.version_repository = version_repository
        self.s3_storage = s3_storage
        self.settings = get_settings()

    def _validate_upload(self, *, filename: str, content_type_header: str | None, size: int) -> ResumeContentType:
        max_bytes = self.settings.RESUME_MAX_UPLOAD_MB * 1024 * 1024
        if size <= 0:
            raise ValidationError("Uploaded file is empty.", error_code="empty_file")
        if size > max_bytes:
            raise ValidationError(
                f"File exceeds the {self.settings.RESUME_MAX_UPLOAD_MB}MB upload limit.",
                error_code="file_too_large",
            )
        return _resolve_content_type(filename, content_type_header)

    async def upload_resume(
        self,
        *,
        user_id: uuid.UUID,
        filename: str,
        content_type_header: str | None,
        data: bytes,
        title: str | None,
    ) -> tuple[Resume, ResumeVersion]:
        content_type = self._validate_upload(
            filename=filename, content_type_header=content_type_header, size=len(data)
        )

        resume = await self.resume_repository.create(user_id=user_id, title=title or filename)
        version = await self._create_version(
            resume=resume,
            filename=filename,
            content_type=content_type,
            data=data,
            version_number=1,
        )
        await self.version_repository.set_current(resume_id=resume.id, version_id=version.id)

        logger.info(
            "Resume uploaded",
            extra={"extra_fields": {"resume_id": str(resume.id), "user_id": str(user_id)}},
        )
        self._enqueue_parsing(version.id)
        return resume, version

    async def upload_new_version(
        self,
        *,
        user_id: uuid.UUID,
        resume_id: uuid.UUID,
        filename: str,
        content_type_header: str | None,
        data: bytes,
    ) -> ResumeVersion:
        resume = await self.get_resume(user_id=user_id, resume_id=resume_id)
        content_type = self._validate_upload(
            filename=filename, content_type_header=content_type_header, size=len(data)
        )

        version_number = await self.version_repository.next_version_number(resume_id=resume.id)
        version = await self._create_version(
            resume=resume,
            filename=filename,
            content_type=content_type,
            data=data,
            version_number=version_number,
        )
        await self.version_repository.set_current(resume_id=resume.id, version_id=version.id)

        logger.info(
            "Resume version uploaded",
            extra={
                "extra_fields": {
                    "resume_id": str(resume.id),
                    "version_id": str(version.id),
                    "version_number": version_number,
                }
            },
        )
        self._enqueue_parsing(version.id)
        return version

    async def _create_version(
        self,
        *,
        resume: Resume,
        filename: str,
        content_type: ResumeContentType,
        data: bytes,
        version_number: int,
    ) -> ResumeVersion:
        version = await self.version_repository.create(
            resume_id=resume.id,
            version_number=version_number,
            is_current=False,
            status=ResumeVersionStatus.UPLOADED,
            content_type=content_type,
            original_filename=filename,
            s3_bucket=self.s3_storage.bucket,
            s3_key="",
            file_size_bytes=len(data),
        )
        key = _s3_key(resume.user_id, resume.id, version.id, filename)
        self.s3_storage.upload_bytes(data, key)
        await self.version_repository.update(version, s3_key=key)
        return version

    def _enqueue_parsing(self, version_id: uuid.UUID) -> None:
        from app.modules.resume_parser.tasks import parse_resume_task

        parse_resume_task.delay(str(version_id))

    async def list_resumes(
        self, *, user_id: uuid.UUID, offset: int, limit: int
    ) -> tuple[Sequence[Resume], int]:
        resumes = await self.resume_repository.list_by_user(user_id=user_id, offset=offset, limit=limit)
        total = await self.resume_repository.count_by_user(user_id=user_id)
        return resumes, total

    async def get_resume(self, *, user_id: uuid.UUID, resume_id: uuid.UUID) -> Resume:
        resume = await self.resume_repository.get_owned(resume_id=resume_id, user_id=user_id)
        if resume is None:
            raise NotFoundError("Resume not found.")
        return resume

    async def get_version(
        self,
        *,
        user_id: uuid.UUID,
        resume_id: uuid.UUID,
        version_id: uuid.UUID,
        with_details: bool = False,
    ) -> ResumeVersion:
        await self.get_resume(user_id=user_id, resume_id=resume_id)
        version = await self.version_repository.get_for_resume(
            resume_id=resume_id, version_id=version_id, with_details=with_details
        )
        if version is None:
            raise NotFoundError("Resume version not found.")
        return version

    async def list_versions(
        self, *, user_id: uuid.UUID, resume_id: uuid.UUID
    ) -> Sequence[ResumeVersion]:
        await self.get_resume(user_id=user_id, resume_id=resume_id)
        return await self.version_repository.list_for_resume(resume_id=resume_id)

    async def update_resume(self, *, user_id: uuid.UUID, resume_id: uuid.UUID, title: str) -> Resume:
        resume = await self.get_resume(user_id=user_id, resume_id=resume_id)
        return await self.resume_repository.update(resume, title=title)

    async def delete_resume(self, *, user_id: uuid.UUID, resume_id: uuid.UUID) -> None:
        resume = await self.get_resume(user_id=user_id, resume_id=resume_id)
        await self.resume_repository.update(resume, deleted_at=datetime.now(timezone.utc))

    async def delete_version(
        self, *, user_id: uuid.UUID, resume_id: uuid.UUID, version_id: uuid.UUID
    ) -> None:
        version = await self.get_version(user_id=user_id, resume_id=resume_id, version_id=version_id)
        remaining = await self.version_repository.list_for_resume(resume_id=resume_id)
        if len(remaining) <= 1:
            raise ConflictError("Cannot delete the only version of a resume.")

        was_current = version.is_current
        await self.version_repository.update(version, deleted_at=datetime.now(timezone.utc))

        if was_current:
            candidates = [v for v in remaining if v.id != version.id]
            newest = max(candidates, key=lambda v: v.version_number)
            await self.version_repository.set_current(resume_id=resume_id, version_id=newest.id)

    async def activate_version(
        self, *, user_id: uuid.UUID, resume_id: uuid.UUID, version_id: uuid.UUID
    ) -> ResumeVersion:
        version = await self.get_version(user_id=user_id, resume_id=resume_id, version_id=version_id)
        await self.version_repository.set_current(resume_id=resume_id, version_id=version_id)
        version.is_current = True
        return version

    async def get_download_url(
        self, *, user_id: uuid.UUID, resume_id: uuid.UUID, version_id: uuid.UUID
    ) -> str:
        version = await self.get_version(user_id=user_id, resume_id=resume_id, version_id=version_id)
        return self.s3_storage.generate_presigned_download_url(version.s3_key)
