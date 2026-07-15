import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { AdminTable } from "@/features/admin/components/AdminTable";
import { StatTile } from "@/features/admin/components/StatTile";
import { StatusBadge } from "@/features/admin/components/StatusBadge";
import { getDashboard } from "@/features/admin/api";

export function DashboardPage() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["admin", "dashboard"],
    queryFn: getDashboard,
  });

  return (
    <section>
      <header className="admin-page__header">
        <h1>Dashboard</h1>
        <p className="admin-page__subtitle">Platform-wide snapshot across users, content, and moderation.</p>
      </header>

      {isLoading && <p className="admin-status">Loading…</p>}
      {isError && <p className="admin-status admin-status--error">Unable to load the dashboard.</p>}

      {data && (
        <>
          <div className="admin-stat-grid">
            <StatTile label="Users" value={data.users_total} hint={`${data.users_active} active`} />
            <StatTile label="Suspended" value={data.users_suspended} />
            <StatTile label="Banned" value={data.users_banned} />
            <StatTile
              label="Companies"
              value={data.companies_total}
              hint={`${data.companies_verified} verified · ${data.companies_pending} pending`}
            />
            <StatTile label="Jobs" value={data.jobs_total} hint={`${data.jobs_flagged} flagged`} />
            <StatTile label="Applications" value={data.applications_total} />
            <StatTile label="Open reports" value={data.reports_open} />
            <StatTile label="Feature flags enabled" value={data.feature_flags_enabled} />
          </div>

          <div className="admin-section">
            <h2 className="admin-section__title">
              Recent audit log <Link className="admin-link" to="/admin/audit-logs">View all</Link>
            </h2>
            <AdminTable
              rows={data.recent_audit_logs}
              rowKey={(row) => row.id}
              emptyMessage="No audit log entries yet."
              columns={[
                { key: "created_at", header: "When", render: (r) => new Date(r.created_at).toLocaleString() },
                { key: "actor", header: "Actor", render: (r) => r.actor_email ?? "system" },
                { key: "action", header: "Action", render: (r) => r.action },
                { key: "target", header: "Target", render: (r) => r.target_type },
              ]}
            />
          </div>

          <div className="admin-section">
            <h2 className="admin-section__title">
              Recent system events <Link className="admin-link" to="/admin/system-logs">View all</Link>
            </h2>
            <AdminTable
              rows={data.recent_system_logs}
              rowKey={(row) => `${row.source}-${row.occurred_at}-${row.target_id ?? ""}`}
              emptyMessage="No system events recorded."
              columns={[
                { key: "occurred_at", header: "When", render: (r) => new Date(r.occurred_at).toLocaleString() },
                { key: "severity", header: "Severity", render: (r) => <StatusBadge status={r.severity} /> },
                { key: "source", header: "Source", render: (r) => r.source },
                { key: "message", header: "Message", render: (r) => r.message },
              ]}
            />
          </div>
        </>
      )}
    </section>
  );
}
