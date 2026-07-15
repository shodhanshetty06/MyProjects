import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState, type FormEvent } from "react";
import { Link, useParams } from "react-router-dom";

import { getUser, updateUserStatus } from "@/features/admin/api";
import { StatusBadge } from "@/features/admin/components/StatusBadge";
import type { UserAccountStatus } from "@/features/admin/types";

export function UserDetailPage() {
  const { userId } = useParams<{ userId: string }>();
  const queryClient = useQueryClient();
  const [status, setStatus] = useState<UserAccountStatus>("active");
  const [reason, setReason] = useState("");

  const { data, isLoading, isError } = useQuery({
    queryKey: ["admin", "users", userId],
    queryFn: () => getUser(userId as string),
    enabled: Boolean(userId),
  });

  useEffect(() => {
    if (data) {
      setStatus(data.status);
    }
  }, [data]);

  const mutation = useMutation({
    mutationFn: () => updateUserStatus(userId as string, { status, reason: reason.trim() || undefined }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "users"] });
    },
  });

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    mutation.mutate();
  };

  return (
    <section>
      <header className="admin-page__header">
        <Link className="admin-link" to="/admin/users">
          &larr; Back to Users
        </Link>
        <h1 className="admin-mono">{userId}</h1>
      </header>

      {isLoading && <p className="admin-status">Loading…</p>}
      {isError && <p className="admin-status admin-status--error">Unable to load this user.</p>}

      {data && (
        <>
          <div className="admin-card admin-section">
            <dl className="admin-kv">
              <dt>Status</dt>
              <dd><StatusBadge status={data.status} /></dd>
              <dt>Reason</dt>
              <dd>{data.reason ?? "—"}</dd>
              <dt>Applications</dt>
              <dd>{data.application_count}</dd>
              <dt>Auto-apply tasks</dt>
              <dd>{data.auto_apply_task_count}</dd>
              <dt>Notifications sent</dt>
              <dd>{data.notification_count}</dd>
              <dt>First seen</dt>
              <dd>{data.first_seen_at ? new Date(data.first_seen_at).toLocaleString() : "—"}</dd>
              <dt>Last activity</dt>
              <dd>{data.last_activity_at ? new Date(data.last_activity_at).toLocaleString() : "—"}</dd>
            </dl>
          </div>

          <div className="admin-card admin-section">
            <h2 className="admin-section__title">Applications by status</h2>
            {Object.keys(data.applications_by_status).length === 0 ? (
              <p className="admin-empty">No applications.</p>
            ) : (
              <ul>
                {Object.entries(data.applications_by_status).map(([key, count]) => (
                  <li key={key}>
                    {key}: {count}
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="admin-card">
            <h2 className="admin-section__title">Update account status</h2>
            <form onSubmit={handleSubmit} className="admin-form-grid">
              <div className="admin-form-row">
                <label htmlFor="status-select">Status</label>
                <select
                  id="status-select"
                  className="admin-select"
                  value={status}
                  onChange={(event) => setStatus(event.target.value as UserAccountStatus)}
                >
                  <option value="active">Active</option>
                  <option value="suspended">Suspended</option>
                  <option value="banned">Banned</option>
                </select>
              </div>
              <div className="admin-form-row">
                <label htmlFor="status-reason">Reason (optional)</label>
                <input
                  id="status-reason"
                  className="admin-text-input"
                  value={reason}
                  onChange={(event) => setReason(event.target.value)}
                />
              </div>
              <button type="submit" className="admin-button" disabled={mutation.isPending}>
                {mutation.isPending ? "Saving…" : "Update status"}
              </button>
            </form>
            {mutation.isError && <p className="admin-status admin-status--error">Update failed.</p>}
            {mutation.isSuccess && <p className="admin-status">Status updated.</p>}
          </div>
        </>
      )}
    </section>
  );
}
