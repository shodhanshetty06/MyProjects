import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { listSystemLogs } from "@/features/admin/api";
import { AdminTable } from "@/features/admin/components/AdminTable";
import { Pagination } from "@/features/admin/components/Pagination";
import { StatusBadge } from "@/features/admin/components/StatusBadge";
import type { SystemLogFiltersParams } from "@/features/admin/types";

export function SystemLogsPage() {
  const [source, setSource] = useState("");
  const [severity, setSeverity] = useState("");
  const [page, setPage] = useState(1);

  const filters: SystemLogFiltersParams = useMemo(
    () => ({ source: source.trim() || undefined, severity: severity || undefined, page, page_size: 25 }),
    [source, severity, page],
  );

  const { data, isLoading, isError } = useQuery({
    queryKey: ["admin", "system-logs", filters],
    queryFn: () => listSystemLogs(filters),
    placeholderData: keepPreviousData,
  });

  return (
    <section>
      <header className="admin-page__header">
        <h1>System Logs</h1>
        <p className="admin-page__subtitle">
          Failure/error signals aggregated across ingestion, auto-apply, email tracking, and
          notifications — a read-only feed, no structured log store exists yet.
        </p>
      </header>

      <div className="admin-toolbar">
        <input
          type="search"
          className="admin-text-input"
          placeholder="Filter by source"
          aria-label="Filter by source"
          value={source}
          onChange={(event) => {
            setPage(1);
            setSource(event.target.value);
          }}
        />
        <select
          className="admin-select"
          aria-label="Severity"
          value={severity}
          onChange={(event) => {
            setPage(1);
            setSeverity(event.target.value);
          }}
        >
          <option value="">All severities</option>
          <option value="error">Error</option>
          <option value="warning">Warning</option>
        </select>
      </div>

      {isLoading && <p className="admin-status">Loading…</p>}
      {isError && <p className="admin-status admin-status--error">Unable to load system logs.</p>}

      {data && (
        <>
          <AdminTable
            rows={data.items}
            rowKey={(row) => `${row.source}-${row.occurred_at}-${row.target_id ?? ""}`}
            emptyMessage="No system events match these filters."
            columns={[
              { key: "occurred_at", header: "When", render: (r) => new Date(r.occurred_at).toLocaleString() },
              { key: "severity", header: "Severity", render: (r) => <StatusBadge status={r.severity} /> },
              { key: "source", header: "Source", render: (r) => r.source },
              { key: "message", header: "Message", render: (r) => r.message },
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
