import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { deleteApplication, listApplications } from "@/features/admin/api";
import { AdminTable } from "@/features/admin/components/AdminTable";
import { Pagination } from "@/features/admin/components/Pagination";
import { StatusBadge } from "@/features/admin/components/StatusBadge";
import { useHasPermission } from "@/features/admin/context";
import type { ApplicationFiltersParams, ApplicationStatus } from "@/features/admin/types";

export function ApplicationsPage() {
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState<ApplicationStatus | "">("");
  const [page, setPage] = useState(1);
  const canManage = useHasPermission("users.manage");
  const queryClient = useQueryClient();

  const filters: ApplicationFiltersParams = useMemo(
    () => ({ search: search.trim() || undefined, status: status || undefined, page, page_size: 20 }),
    [search, status, page],
  );

  const { data, isLoading, isError } = useQuery({
    queryKey: ["admin", "applications", filters],
    queryFn: () => listApplications(filters),
    placeholderData: keepPreviousData,
  });

  const deleteMutation = useMutation({
    mutationFn: (applicationId: string) => deleteApplication(applicationId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "applications"] }),
  });

  return (
    <section>
      <header className="admin-page__header">
        <h1>Applications</h1>
        <p className="admin-page__subtitle">Platform-wide application records, for support and abuse cleanup.</p>
      </header>

      <div className="admin-toolbar">
        <input
          type="search"
          className="admin-text-input"
          placeholder="Search by company or job title"
          aria-label="Search"
          value={search}
          onChange={(event) => {
            setPage(1);
            setSearch(event.target.value);
          }}
        />
        <select
          className="admin-select"
          aria-label="Status"
          value={status}
          onChange={(event) => {
            setPage(1);
            setStatus(event.target.value as ApplicationStatus | "");
          }}
        >
          <option value="">All statuses</option>
          <option value="applied">Applied</option>
          <option value="interview">Interview</option>
          <option value="assessment">Assessment</option>
          <option value="offer">Offer</option>
          <option value="accepted">Accepted</option>
          <option value="rejected">Rejected</option>
          <option value="withdrawn">Withdrawn</option>
        </select>
      </div>

      {isLoading && <p className="admin-status">Loading…</p>}
      {isError && <p className="admin-status admin-status--error">Unable to load applications.</p>}

      {data && (
        <>
          <AdminTable
            rows={data.items}
            rowKey={(row) => row.id}
            emptyMessage="No applications found."
            columns={[
              { key: "company", header: "Company", render: (r) => r.company_name },
              { key: "title", header: "Job title", render: (r) => r.job_title },
              { key: "user", header: "User", render: (r) => <span className="admin-mono">{r.user_id}</span> },
              { key: "status", header: "Status", render: (r) => <StatusBadge status={r.status} /> },
              { key: "applied_at", header: "Applied", render: (r) => new Date(r.applied_at).toLocaleDateString() },
              {
                key: "actions",
                header: "",
                render: (r) =>
                  canManage && (
                    <button
                      type="button"
                      className="admin-button admin-button--danger"
                      disabled={deleteMutation.isPending}
                      onClick={() => deleteMutation.mutate(r.id)}
                    >
                      Delete
                    </button>
                  ),
              },
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
