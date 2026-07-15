import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { getModerationQueue } from "@/features/admin/api";
import { AdminTable } from "@/features/admin/components/AdminTable";
import { StatusBadge } from "@/features/admin/components/StatusBadge";

const LINK_BY_KIND: Record<string, string> = {
  report: "/admin/reports",
  job: "/admin/jobs",
  company: "/admin/companies",
};

export function ModerationQueuePage() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["admin", "moderation", "queue"],
    queryFn: getModerationQueue,
  });

  return (
    <section>
      <header className="admin-page__header">
        <h1>Content Moderation</h1>
        <p className="admin-page__subtitle">
          Everything awaiting review: open reports, flagged/pending jobs, and pending company
          verifications, newest first.
        </p>
      </header>

      {isLoading && <p className="admin-status">Loading…</p>}
      {isError && <p className="admin-status admin-status--error">Unable to load the moderation queue.</p>}

      {data && (
        <AdminTable
          rows={data.items}
          rowKey={(row) => `${row.kind}-${row.id}`}
          emptyMessage="Nothing pending review."
          columns={[
            { key: "kind", header: "Kind", render: (r) => <StatusBadge status={r.kind} /> },
            { key: "summary", header: "Summary", render: (r) => r.summary },
            { key: "status", header: "Status", render: (r) => <StatusBadge status={r.status} /> },
            { key: "created_at", header: "Filed", render: (r) => new Date(r.created_at).toLocaleString() },
            {
              key: "actions",
              header: "",
              render: (r) => (
                <Link className="admin-link" to={LINK_BY_KIND[r.kind] ?? "/admin"}>
                  Review
                </Link>
              ),
            },
          ]}
        />
      )}
    </section>
  );
}
