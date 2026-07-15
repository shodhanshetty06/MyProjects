import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { listReports, updateReport } from "@/features/admin/api";
import { AdminTable } from "@/features/admin/components/AdminTable";
import { Pagination } from "@/features/admin/components/Pagination";
import { StatusBadge } from "@/features/admin/components/StatusBadge";
import { useHasPermission } from "@/features/admin/context";
import type { ReportFiltersParams, ReportResponse, ReportStatus, ReportTargetType } from "@/features/admin/types";

function ResolveAction({ report }: { report: ReportResponse }) {
  const canModerate = useHasPermission("content.moderate");
  const queryClient = useQueryClient();
  const [status, setStatus] = useState<ReportStatus>(
    report.status === "open" ? "reviewing" : report.status,
  );
  const [notes, setNotes] = useState("");

  const mutation = useMutation({
    mutationFn: () => updateReport(report.id, { status, resolution_notes: notes.trim() || undefined }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "reports"] }),
  });

  if (!canModerate || report.status === "resolved" || report.status === "dismissed") {
    return <StatusBadge status={report.status} />;
  }

  return (
    <div className="admin-inline-form">
      <select
        className="admin-select"
        aria-label={`Status for report ${report.id}`}
        value={status}
        onChange={(event) => setStatus(event.target.value as ReportStatus)}
      >
        <option value="open">Open</option>
        <option value="reviewing">Reviewing</option>
        <option value="resolved">Resolved</option>
        <option value="dismissed">Dismissed</option>
      </select>
      <input
        className="admin-text-input"
        placeholder="Notes"
        aria-label="Resolution notes"
        value={notes}
        onChange={(event) => setNotes(event.target.value)}
      />
      <button
        type="button"
        className="admin-button"
        disabled={mutation.isPending}
        onClick={() => mutation.mutate()}
      >
        {mutation.isPending ? "Saving…" : "Apply"}
      </button>
    </div>
  );
}

export function ReportsPage() {
  const [status, setStatus] = useState<ReportStatus | "">("");
  const [targetType, setTargetType] = useState<ReportTargetType | "">("");
  const [page, setPage] = useState(1);

  const filters: ReportFiltersParams = useMemo(
    () => ({ status: status || undefined, target_type: targetType || undefined, page, page_size: 20 }),
    [status, targetType, page],
  );

  const { data, isLoading, isError } = useQuery({
    queryKey: ["admin", "reports", filters],
    queryFn: () => listReports(filters),
    placeholderData: keepPreviousData,
  });

  return (
    <section>
      <header className="admin-page__header">
        <h1>Reports</h1>
        <p className="admin-page__subtitle">User-filed reports against jobs, companies, or other users.</p>
      </header>

      <div className="admin-toolbar">
        <select
          className="admin-select"
          aria-label="Status"
          value={status}
          onChange={(event) => {
            setPage(1);
            setStatus(event.target.value as ReportStatus | "");
          }}
        >
          <option value="">All statuses</option>
          <option value="open">Open</option>
          <option value="reviewing">Reviewing</option>
          <option value="resolved">Resolved</option>
          <option value="dismissed">Dismissed</option>
        </select>
        <select
          className="admin-select"
          aria-label="Target type"
          value={targetType}
          onChange={(event) => {
            setPage(1);
            setTargetType(event.target.value as ReportTargetType | "");
          }}
        >
          <option value="">All target types</option>
          <option value="job">Job</option>
          <option value="company">Company</option>
          <option value="user">User</option>
        </select>
      </div>

      {isLoading && <p className="admin-status">Loading…</p>}
      {isError && <p className="admin-status admin-status--error">Unable to load reports.</p>}

      {data && (
        <>
          <AdminTable
            rows={data.items}
            rowKey={(row) => row.id}
            emptyMessage="No reports found."
            columns={[
              { key: "target", header: "Target", render: (r) => `${r.target_type} · ${r.target_id.slice(0, 8)}` },
              { key: "reason", header: "Reason", render: (r) => r.reason },
              { key: "created_at", header: "Filed", render: (r) => new Date(r.created_at).toLocaleString() },
              { key: "status", header: "Status", render: (r) => <ResolveAction report={r} /> },
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
