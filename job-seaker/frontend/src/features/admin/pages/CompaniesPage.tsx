import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { listCompanies, updateCompanyVerification } from "@/features/admin/api";
import { AdminTable } from "@/features/admin/components/AdminTable";
import { Pagination } from "@/features/admin/components/Pagination";
import { StatusBadge } from "@/features/admin/components/StatusBadge";
import { useHasPermission } from "@/features/admin/context";
import type { CompanyFiltersParams, CompanySummary, VerificationStatus } from "@/features/admin/types";

function VerificationAction({ company }: { company: CompanySummary }) {
  const canVerify = useHasPermission("companies.verify");
  const queryClient = useQueryClient();
  const [status, setStatus] = useState<VerificationStatus>(company.verification_status);
  const [notes, setNotes] = useState("");

  const mutation = useMutation({
    mutationFn: () => updateCompanyVerification(company.id, { status, notes: notes.trim() || undefined }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "companies"] }),
  });

  if (!canVerify) {
    return <StatusBadge status={company.verification_status} />;
  }

  return (
    <div className="admin-inline-form">
      <select
        className="admin-select"
        aria-label={`Verification status for ${company.name}`}
        value={status}
        onChange={(event) => setStatus(event.target.value as VerificationStatus)}
      >
        <option value="unverified">Unverified</option>
        <option value="pending">Pending</option>
        <option value="verified">Verified</option>
        <option value="rejected">Rejected</option>
      </select>
      <input
        className="admin-text-input"
        placeholder="Notes"
        aria-label={`Notes for ${company.name}`}
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

export function CompaniesPage() {
  const [searchParams] = useSearchParams();
  const [search, setSearch] = useState("");
  const [verificationStatus, setVerificationStatus] = useState<VerificationStatus | "">(
    (searchParams.get("verification_status") as VerificationStatus | null) ?? "",
  );
  const [page, setPage] = useState(1);

  const filters: CompanyFiltersParams = useMemo(
    () => ({
      search: search.trim() || undefined,
      verification_status: verificationStatus || undefined,
      page,
      page_size: 20,
    }),
    [search, verificationStatus, page],
  );

  const { data, isLoading, isError } = useQuery({
    queryKey: ["admin", "companies", filters],
    queryFn: () => listCompanies(filters),
    placeholderData: keepPreviousData,
  });

  return (
    <section>
      <header className="admin-page__header">
        <h1>Companies</h1>
        <p className="admin-page__subtitle">Review and verify employer accounts.</p>
      </header>

      <div className="admin-toolbar">
        <input
          type="search"
          className="admin-text-input"
          placeholder="Search by name"
          aria-label="Search by name"
          value={search}
          onChange={(event) => {
            setPage(1);
            setSearch(event.target.value);
          }}
        />
        <select
          className="admin-select"
          aria-label="Verification status"
          value={verificationStatus}
          onChange={(event) => {
            setPage(1);
            setVerificationStatus(event.target.value as VerificationStatus | "");
          }}
        >
          <option value="">All verification statuses</option>
          <option value="unverified">Unverified</option>
          <option value="pending">Pending</option>
          <option value="verified">Verified</option>
          <option value="rejected">Rejected</option>
        </select>
      </div>

      {isLoading && <p className="admin-status">Loading…</p>}
      {isError && <p className="admin-status admin-status--error">Unable to load companies.</p>}

      {data && (
        <>
          <AdminTable
            rows={data.items}
            rowKey={(row) => row.id}
            emptyMessage="No companies found."
            columns={[
              { key: "name", header: "Name", render: (r) => r.name },
              { key: "industry", header: "Industry", render: (r) => r.industry ?? "—" },
              { key: "jobs", header: "Jobs", align: "right", render: (r) => r.jobs_count },
              { key: "active", header: "Active", render: (r) => (r.is_active ? "Yes" : "No") },
              { key: "verification", header: "Verification", render: (r) => <VerificationAction company={r} /> },
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
