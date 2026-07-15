"""Generic base repository implementing common CRUD operations.

Module-specific repositories subclass `BaseRepository[ModelType]` and only
add query methods that are genuinely specific to that domain — everything
generic (get, list, create, update, delete) is inherited for free.
"""
from __future__ import annotations

import uuid
from typing import Any, Generic, Sequence, Type, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base_class import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Generic async CRUD repository over a single SQLAlchemy model."""

    model: Type[ModelType]

    def __init__(self, session: AsyncSession, model: Type[ModelType] | None = None) -> None:
        self.session = session
        if model is not None:
            self.model = model

    async def get(self, id_: uuid.UUID) -> ModelType | None:
        return await self.session.get(self.model, id_)

    async def list(
        self, *, offset: int = 0, limit: int = 20, **filters: Any
    ) -> Sequence[ModelType]:
        stmt = select(self.model)
        for field, value in filters.items():
            stmt = stmt.where(getattr(self.model, field) == value)
        stmt = stmt.offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def count(self, **filters: Any) -> int:
        from sqlalchemy import func

        stmt = select(func.count()).select_from(self.model)
        for field, value in filters.items():
            stmt = stmt.where(getattr(self.model, field) == value)
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def create(self, **kwargs: Any) -> ModelType:
        instance = self.model(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        return instance

    async def update(self, instance: ModelType, **kwargs: Any) -> ModelType:
        for field, value in kwargs.items():
            setattr(instance, field, value)
        await self.session.flush()
        return instance

    async def delete(self, instance: ModelType) -> None:
        await self.session.delete(instance)
        await self.session.flush()
