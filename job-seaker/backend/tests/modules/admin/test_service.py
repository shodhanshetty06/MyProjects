"""DB-free unit tests for the admin module's static RBAC table and filter
schemas — no Postgres required, run with plain `pytest` like
`tests/modules/search/test_scoring.py`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.modules.admin.models import AdminRole
from app.modules.admin.permissions import ROLE_PERMISSIONS, Permission, permissions_for_role
from app.modules.admin.schemas import AuditLogFilters, UserFilters


def test_every_role_has_a_permission_set() -> None:
    for role in AdminRole:
        assert role in ROLE_PERMISSIONS


def test_super_admin_has_every_permission() -> None:
    assert permissions_for_role(AdminRole.SUPER_ADMIN) == frozenset(Permission)


def test_admin_lacks_only_permissions_manage() -> None:
    perms = permissions_for_role(AdminRole.ADMIN)
    assert Permission.PERMISSIONS_MANAGE not in perms
    assert Permission.USERS_MANAGE in perms
    assert Permission.FEATURE_FLAGS_MANAGE in perms


def test_analyst_is_read_only() -> None:
    perms = permissions_for_role(AdminRole.ANALYST)
    assert Permission.ANALYTICS_VIEW in perms
    assert Permission.USERS_MANAGE not in perms
    assert Permission.JOBS_MODERATE not in perms


def test_unknown_role_has_no_permissions() -> None:
    assert permissions_for_role("not-a-role") == frozenset()


def test_user_filters_default_pagination() -> None:
    filters = UserFilters()
    assert filters.page == 1
    assert filters.page_size == 20
    assert filters.offset == 0


def test_user_filters_page_size_is_bounded() -> None:
    with pytest.raises(ValidationError):
        UserFilters(page_size=1000)


def test_audit_log_filters_rejects_inverted_date_range() -> None:
    with pytest.raises(ValidationError):
        AuditLogFilters(date_from="2026-02-01T00:00:00Z", date_to="2026-01-01T00:00:00Z")
