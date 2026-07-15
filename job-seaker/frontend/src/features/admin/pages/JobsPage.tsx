import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { listJobs, updateJobModeration } from "@/features/admin/api";
import { AdminTable } from "@/features/admin/components/AdminTable";
import { Pagination } from "@/features/admin/components/Pagination";
import { StatusBadge } from "@/features/admin/components/StatusBadge";
import { useHasPermission } from "@/features/admin/context";
import type { JobFiltersParams, JobSummary, ModerationStatus } from "@/features/admin/types";

function ModerationAction({ job }: { job: JobSummary }) {
  const canModerate = useHasPermission("jobs.moderate");
  const queryClient = useQueryClient();
  const [status, setStatus] = useState<ModerationStatus>(job.moderation_status);
  const [reason, setReason] = useState("");

  const mutation = useMutation({
    mutationFn: () => updateJobModeration(job.id, { status, reason: reason.trim() || undefined }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "jobs"] }),
  });

  if (!canModerate) {
    return <StatusBadge status={job.moderation_status} />;
  }

  return (
    <div className="admin-inline-form">
      <select
        className="admin-select"
        aria-label={`Moderation status for ${job.title}`}
        value={status}
        onChange={(event) => setStatus(event.target.value as ModerationStatus)}
      >
        <option value="active">Active</option>
        <option value="pending_review">Pending review</option>
        <option value="flagged">Flagged</option>
        <option value="removed">Removed</option>
      </select>
      <input
        className="admin-text-input"
        placeholder="Reason"
        aria-label={`Reason for ${job.title}`}
        value={reason}
        onChange={(event) => setReason(event.target.value)}
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

export function JobsPage() {
  const [searchParams] = useSearchParams();
  const [search, setSearch] = useState("");
  const [moderationStatus, setModerationStatus] = useState<ModerationStatus | "">(
    (searchParams.get("moderation_status") as ModerationStatus | null) ?? "",
  );
  const [page, setPage] = useState(1);

  const filters: JobFiltersParams = useMemo(
    () => ({
      search: search.trim() || undefined,
      moderation_status: moderationStatus || undefined,
      page,
      page_size: 20,
    }),
    [search, moderationStatus, page],
  );

  const { data, isLoading, isError } = useQuery({
    queryKey: ["admin", "jobs", filters],
    queryFn: () => listJobs(filters),
    placeholderData: keepPreviousData,
  });

  return (
    <section>
      <header className="admin-page__header">
        <h1>Jobs</h1>
        <p className="admin-page__subtitle">Review job postings and moderate abusive or fraudulent listings.</p>
      </header>

      <div className="admin-toolbar">
        <input
          type="search"
          className="admin-text-input"
          placeholder="Search by title"
          aria-label="Search by title"
          value={search}
          onChange={(event) => {
            setPage(1);
            setSearch(event.target.value);
          }}
        />
        <select
          className="admin-select"
          aria-label="Moderation status"
          value={moderationStatus}
          onChange={(event) => {
            setPage(1);
            setModerationStatus(event.target.value as ModerationStatus | "");
          }}
        >
          <option value="">All moderation statuses</option>
          <option value="active">Active</option>
          <option value="pending_review">Pending review</option>
          <option value="flagged">Flagged</option>
          <option value="removed">Removed</option>
        </select>
      </div>

      {isLoading && <p className="admin-status">Loading…</p>}
      {isError && <p className="admin-status admin-status--error">Unable to load jobs.</p>}

      {data && (
        <>
          <AdminTable
            rows={data.items}
            rowKey={(row) => row.id}
            emptyMessage="No jobs found."
            columns={[
              { key: "title", header: "Title", render: (r) => r.title },
              { key: "company", header: "Company", render: (r) => r.company_name },
              { key: "status", header: "Status", render: (r) => <StatusBadge status={r.status} /> },
              { key: "flags", header: "Flags", align: "right", render: (r) => r.flag_count },
              { key: "moderation", header: "Moderation", render: (r) => <ModerationAction job={r} /> },
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
