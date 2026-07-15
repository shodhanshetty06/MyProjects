"""Generic base service implementing common application/use-case operations.

Services own business logic and orchestrate one or more repositories. This
base class only wires the repository in and forwards the generic CRUD
operations — module services add real use-case methods on top in later
phases.
"""
from __future__ import annotations

import uuid
from typing import Any, Generic, Sequence, TypeVar

from app.core.exceptions import NotFoundError
from app.repositories.base import BaseRepository, ModelType

RepositoryType = TypeVar("RepositoryType", bound=BaseRepository)


class BaseService(Generic[ModelType]):
    """Generic async service delegating persistence to a BaseRepository."""

    not_found_message = "Resource not found."

    def __init__(self, repository: BaseRepository[ModelType]) -> None:
        self.repository = repository

    async def get(self, id_: uuid.UUID) -> ModelType:
        instance = await self.repository.get(id_)
        if instance is None:
            raise NotFoundError(self.not_found_message)
        return instance

    async def list(
        self, *, offset: int = 0, limit: int = 20, **filters: Any
    ) -> Sequence[ModelType]:
        return await self.repository.list(offset=offset, limit=limit, **filters)

    async def count(self, **filters: Any) -> int:
        return await self.repository.count(**filters)

    async def create(self, **kwargs: Any) -> ModelType:
        return await self.repository.create(**kwargs)

    async def update(self, id_: uuid.UUID, **kwargs: Any) -> ModelType:
        instance = await self.get(id_)
        return await self.repository.update(instance, **kwargs)

    async def delete(self, id_: uuid.UUID) -> None:
        instance = await self.get(id_)
        await self.repository.delete(instance)
