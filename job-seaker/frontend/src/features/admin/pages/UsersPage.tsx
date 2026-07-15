import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { listUsers } from "@/features/admin/api";
import { AdminTable } from "@/features/admin/components/AdminTable";
import { Pagination } from "@/features/admin/components/Pagination";
import { StatusBadge } from "@/features/admin/components/StatusBadge";
import type { UserAccountStatus, UserFiltersParams } from "@/features/admin/types";

export function UsersPage() {
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState<UserAccountStatus | "">("");
  const [page, setPage] = useState(1);

  const filters: UserFiltersParams = useMemo(
    () => ({ search: search.trim() || undefined, status: status || undefined, page, page_size: 20 }),
    [search, status, page],
  );

  const { data, isLoading, isError } = useQuery({
    queryKey: ["admin", "users", filters],
    queryFn: () => listUsers(filters),
    placeholderData: keepPreviousData,
  });

  return (
    <section>
      <header className="admin-page__header">
        <h1>Users</h1>
        <p className="admin-page__subtitle">
          Aggregated from application activity — there is no dedicated account table yet, so users
          are identified by id only.
        </p>
      </header>

      <div className="admin-toolbar">
        <input
          type="search"
          className="admin-text-input"
          placeholder="Search by user id"
          aria-label="Search by user id"
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
            setStatus(event.target.value as UserAccountStatus | "");
          }}
        >
          <option value="">All statuses</option>
          <option value="active">Active</option>
          <option value="suspended">Suspended</option>
          <option value="banned">Banned</option>
        </select>
      </div>

      {isLoading && <p className="admin-status">Loading…</p>}
      {isError && <p className="admin-status admin-status--error">Unable to load users.</p>}

      {data && (
        <>
          <AdminTable
            rows={data.items}
            rowKey={(row) => row.user_id}
            emptyMessage="No users found."
            columns={[
              {
                key: "user_id",
                header: "User ID",
                render: (r) => (
                  <Link className="admin-link admin-mono" to={`/admin/users/${r.user_id}`}>
                    {r.user_id}
                  </Link>
                ),
              },
              { key: "status", header: "Status", render: (r) => <StatusBadge status={r.status} /> },
              { key: "applications", header: "Applications", align: "right", render: (r) => r.application_count },
              {
                key: "last_activity",
                header: "Last activity",
                render: (r) => (r.last_activity_at ? new Date(r.last_activity_at).toLocaleString() : "—"),
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
