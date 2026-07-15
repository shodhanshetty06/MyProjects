"""Admin module — static role-based access control.

Roles are fixed (`AdminRole`, in `models.py`); permissions are a fixed set of
capability flags checked by `deps.require_permission`. There is no dynamic
per-actor permission grant in this phase — role membership alone determines
`ROLE_PERMISSIONS`.
"""

from __future__ import annotations

import enum

from app.modules.admin.models import AdminRole


class Permission(str, enum.Enum):
    DASHBOARD_VIEW = "dashboard.view"
    USERS_MANAGE = "users.manage"
    COMPANIES_VERIFY = "companies.verify"
    JOBS_MODERATE = "jobs.moderate"
    CONTENT_MODERATE = "content.moderate"
    REPORTS_MANAGE = "reports.manage"
    FEATURE_FLAGS_MANAGE = "feature_flags.manage"
    AUDIT_LOGS_VIEW = "audit_logs.view"
    SYSTEM_LOGS_VIEW = "system_logs.view"
    ANALYTICS_VIEW = "analytics.view"
    PERMISSIONS_MANAGE = "permissions.manage"


_ALL_PERMISSIONS = frozenset(Permission)

ROLE_PERMISSIONS: dict[AdminRole, frozenset[Permission]] = {
    AdminRole.SUPER_ADMIN: _ALL_PERMISSIONS,
    AdminRole.ADMIN: _ALL_PERMISSIONS - {Permission.PERMISSIONS_MANAGE},
    AdminRole.MODERATOR: frozenset(
        {
            Permission.DASHBOARD_VIEW,
            Permission.COMPANIES_VERIFY,
            Permission.JOBS_MODERATE,
            Permission.CONTENT_MODERATE,
            Permission.REPORTS_MANAGE,
        }
    ),
    AdminRole.SUPPORT: frozenset(
        {
            Permission.DASHBOARD_VIEW,
            Permission.USERS_MANAGE,
        }
    ),
    AdminRole.ANALYST: frozenset(
        {
            Permission.DASHBOARD_VIEW,
            Permission.ANALYTICS_VIEW,
            Permission.SYSTEM_LOGS_VIEW,
            Permission.AUDIT_LOGS_VIEW,
        }
    ),
}


def permissions_for_role(role: AdminRole) -> frozenset[Permission]:
    return ROLE_PERMISSIONS.get(role, frozenset())


__all__ = ["Permission", "ROLE_PERMISSIONS", "permissions_for_role"]
