import { useQuery } from "@tanstack/react-query";
import { isAxiosError } from "axios";
import { useState, type FormEvent } from "react";
import { NavLink, Outlet } from "react-router-dom";

import "@/features/admin/admin.css";
import { bootstrapAdminActor, getPermissions } from "@/features/admin/api";
import { AdminAuthProvider } from "@/features/admin/context";
import type { Permission } from "@/features/admin/types";

interface NavItem {
  to: string;
  label: string;
  permission?: Permission;
  end?: boolean;
}

interface NavGroup {
  label: string;
  items: NavItem[];
}

const NAV_GROUPS: NavGroup[] = [
  { label: "", items: [{ to: "/admin", label: "Dashboard", permission: "dashboard.view", end: true }] },
  {
    label: "Manage",
    items: [
      { to: "/admin/users", label: "Users", permission: "users.manage" },
      { to: "/admin/companies", label: "Companies", permission: "dashboard.view" },
      { to: "/admin/jobs", label: "Jobs", permission: "dashboard.view" },
      { to: "/admin/applications", label: "Applications", permission: "users.manage" },
      { to: "/admin/reports", label: "Reports", permission: "content.moderate" },
    ],
  },
  {
    label: "Moderation",
    items: [
      { to: "/admin/moderation", label: "Content Moderation", permission: "content.moderate" },
      { to: "/admin/jobs?moderation_status=flagged", label: "Job Moderation", permission: "dashboard.view" },
      {
        to: "/admin/companies?verification_status=pending",
        label: "Company Verification",
        permission: "dashboard.view",
      },
    ],
  },
  {
    label: "Insights",
    items: [
      { to: "/admin/analytics", label: "Analytics", permission: "analytics.view" },
      { to: "/admin/system-logs", label: "System Logs", permission: "system_logs.view" },
      { to: "/admin/audit-logs", label: "Audit Logs", permission: "audit_logs.view" },
    ],
  },
  {
    label: "System",
    items: [
      { to: "/admin/feature-flags", label: "Feature Flags", permission: "dashboard.view" },
      { to: "/admin/permissions", label: "Permissions" },
    ],
  },
];

function BootstrapGate() {
  const [email, setEmail] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setIsSubmitting(true);
    setError(null);
    try {
      await bootstrapAdminActor({ email, display_name: displayName || undefined });
      window.location.reload();
    } catch (err) {
      if (isAxiosError(err) && err.response?.status === 409) {
        setError(
          "An admin actor already exists. Ask an existing SUPER_ADMIN/ADMIN to create your account via Permissions.",
        );
      } else {
        setError("Could not bootstrap an admin account. Please try again.");
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="admin-shell">
      <div className="admin-content">
        <div className="admin-bootstrap">
          <h1>Set up the Admin Portal</h1>
          <p className="admin-page__subtitle">
            No admin account exists for this browser yet. Bootstrap the first account as
            SUPER_ADMIN — this only works once.
          </p>
          <form onSubmit={handleSubmit} className="admin-card">
            <div className="admin-form-row">
              <label htmlFor="bootstrap-email">Email</label>
              <input
                id="bootstrap-email"
                className="admin-text-input"
                type="email"
                required
                value={email}
                onChange={(event) => setEmail(event.target.value)}
              />
            </div>
            <div className="admin-form-row">
              <label htmlFor="bootstrap-name">Display name</label>
              <input
                id="bootstrap-name"
                className="admin-text-input"
                type="text"
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
              />
            </div>
            <button type="submit" className="admin-button" disabled={isSubmitting}>
              {isSubmitting ? "Creating…" : "Become Admin"}
            </button>
            {error && <p className="admin-status admin-status--error">{error}</p>}
          </form>
        </div>
      </div>
    </div>
  );
}

export function AdminLayout() {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["admin", "permissions"],
    queryFn: getPermissions,
    retry: false,
  });

  if (isLoading) {
    return (
      <div className="admin-shell">
        <div className="admin-content">
          <p className="admin-status">Loading admin portal…</p>
        </div>
      </div>
    );
  }

  if (isError) {
    if (isAxiosError(error) && error.response?.status === 403) {
      return <BootstrapGate />;
    }
    return (
      <div className="admin-shell">
        <div className="admin-content">
          <p className="admin-status admin-status--error">Unable to reach the Admin API.</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return null;
  }

  return (
    <AdminAuthProvider value={{ role: data.current_role, permissions: data.current_permissions }}>
      <div className="admin-shell">
        <nav className="admin-sidebar" aria-label="Admin navigation">
          <p className="admin-sidebar__title">Admin Portal</p>
          {NAV_GROUPS.map((group) => (
            <div key={group.label || group.items[0].to}>
              {group.label && <p className="admin-sidebar__group-label">{group.label}</p>}
              {group.items
                .filter((item) => !item.permission || data.current_permissions.includes(item.permission))
                .map((item) => (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    end={item.end}
                    className={({ isActive }) => `admin-sidebar__link${isActive ? " active" : ""}`}
                  >
                    {item.label}
                  </NavLink>
                ))}
            </div>
          ))}
          <p className="admin-sidebar__role">Signed in as {data.current_role.replace(/_/g, " ")}</p>
        </nav>
        <div className="admin-content">
          <Outlet />
        </div>
      </div>
    </AdminAuthProvider>
  );
}
