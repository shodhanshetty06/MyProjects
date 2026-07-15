"""Admin module — actor resolution and permission-check dependencies.

Kept module-local (rather than in the global `app/api/deps.py`) because
"who is an admin" is purely an admin-module concern layered on top of the
verified identity `CurrentUserIdDep` already provides.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.api.deps import CurrentUserIdDep, DBSessionDep
from app.core.exceptions import ForbiddenError
from app.modules.admin.models import AdminActor
from app.modules.admin.permissions import Permission, permissions_for_role
from app.modules.admin.repository import AdminRepository


async def get_current_admin_actor(user_id: CurrentUserIdDep, session: DBSessionDep) -> AdminActor:
    actor = await AdminRepository(session).get_actor_by_user_id(user_id)
    if actor is None or not actor.is_active:
        raise ForbiddenError("This account is not registered as an active admin actor.")
    return actor


CurrentAdminActorDep = Annotated[AdminActor, Depends(get_current_admin_actor)]


def require_permission(permission: Permission):
    async def _check(actor: CurrentAdminActorDep) -> AdminActor:
        if permission not in permissions_for_role(actor.role):
            # `actor.role` is a `String`-backed column — a plain `str` at
            # runtime once loaded via a query, not an enum instance — so no
            # `.value` here (see the same note in repository.py).
            raise ForbiddenError(
                f"Role '{actor.role}' does not have permission '{permission.value}'."
            )
        return actor

    return Depends(_check)


__all__ = ["get_current_admin_actor", "CurrentAdminActorDep", "require_permission"]
