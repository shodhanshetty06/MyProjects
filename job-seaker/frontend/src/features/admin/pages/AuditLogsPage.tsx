import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { listAuditLogs } from "@/features/admin/api";
import { AdminTable } from "@/features/admin/components/AdminTable";
import { Pagination } from "@/features/admin/components/Pagination";
import type { AuditLogFiltersParams } from "@/features/admin/types";

export function AuditLogsPage() {
  const [action, setAction] = useState("");
  const [targetType, setTargetType] = useState("");
  const [page, setPage] = useState(1);

  const filters: AuditLogFiltersParams = useMemo(
    () => ({ action: action.trim() || undefined, target_type: targetType.trim() || undefined, page, page_size: 25 }),
    [action, targetType, page],
  );

  const { data, isLoading, isError } = useQuery({
    queryKey: ["admin", "audit-logs", filters],
    queryFn: () => listAuditLogs(filters),
    placeholderData: keepPreviousData,
  });

  return (
    <section>
      <header className="admin-page__header">
        <h1>Audit Logs</h1>
        <p className="admin-page__subtitle">Every mutating admin action, with before/after state.</p>
      </header>

      <div className="admin-toolbar">
        <input
          type="search"
          className="admin-text-input"
          placeholder="Filter by action (e.g. user.status_update)"
          aria-label="Filter by action"
          value={action}
          onChange={(event) => {
            setPage(1);
            setAction(event.target.value);
          }}
        />
        <input
          type="search"
          className="admin-text-input"
          placeholder="Filter by target type"
          aria-label="Filter by target type"
          value={targetType}
          onChange={(event) => {
            setPage(1);
            setTargetType(event.target.value);
          }}
        />
      </div>

      {isLoading && <p className="admin-status">Loading…</p>}
      {isError && <p className="admin-status admin-status--error">Unable to load audit logs.</p>}

      {data && (
        <>
          <AdminTable
            rows={data.items}
            rowKey={(row) => row.id}
            emptyMessage="No audit log entries match these filters."
            columns={[
              { key: "created_at", header: "When", render: (r) => new Date(r.created_at).toLocaleString() },
              { key: "actor", header: "Actor", render: (r) => r.actor_email ?? "system" },
              { key: "action", header: "Action", render: (r) => r.action },
              { key: "target", header: "Target", render: (r) => `${r.target_type}${r.target_id ? ` · ${r.target_id.slice(0, 8)}` : ""}` },
              { key: "notes", header: "Notes", render: (r) => r.notes ?? "—" },
            ]}
          />
          <Pagination
            page={data.page}
            totalPages={data.totalPages}
            totalItems={data.totalItems}
            onPageChange={setPage}
          />
        </>
      )}
    </section>
  );
}
