import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";

import { createAdminActor, getPermissions, listAdminActors, updateAdminActor } from "@/features/admin/api";
import { AdminTable } from "@/features/admin/components/AdminTable";
import { useHasPermission } from "@/features/admin/context";
import type { AdminActorResponse, AdminRole } from "@/features/admin/types";

const ROLES: AdminRole[] = ["super_admin", "admin", "moderator", "support", "analyst"];

function ActorRoleAction({ actor }: { actor: AdminActorResponse }) {
  const queryClient = useQueryClient();
  const [role, setRole] = useState<AdminRole>(actor.role);

  const roleMutation = useMutation({
    mutationFn: () => updateAdminActor(actor.id, { role }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "actors"] }),
  });
  const activeMutation = useMutation({
    mutationFn: () => updateAdminActor(actor.id, { is_active: !actor.is_active }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "actors"] }),
  });

  return (
    <div className="admin-inline-form">
      <select
        className="admin-select"
        aria-label={`Role for ${actor.email}`}
        value={role}
        onChange={(event) => setRole(event.target.value as AdminRole)}
      >
        {ROLES.map((r) => (
          <option key={r} value={r}>
            {r.replace(/_/g, " ")}
          </option>
        ))}
      </select>
      <button type="button" className="admin-button" disabled={roleMutation.isPending} onClick={() => roleMutation.mutate()}>
        Save role
      </button>
      <button
        type="button"
        className={`admin-button ${actor.is_active ? "admin-button--danger" : "admin-button--ghost"}`}
        disabled={activeMutation.isPending}
        onClick={() => activeMutation.mutate()}
      >
        {actor.is_active ? "Deactivate" : "Activate"}
      </button>
    </div>
  );
}

export function PermissionsPage() {
  const canManage = useHasPermission("permissions.manage");
  const queryClient = useQueryClient();
  const [userId, setUserId] = useState("");
  const [email, setEmail] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [role, setRole] = useState<AdminRole>("support");

  const { data: permissions, isLoading: permissionsLoading } = useQuery({
    queryKey: ["admin", "permissions"],
    queryFn: getPermissions,
  });

  const { data: actors, isLoading: actorsLoading } = useQuery({
    queryKey: ["admin", "actors"],
    queryFn: () => listAdminActors(1, 50),
    enabled: canManage,
  });

  const createMutation = useMutation({
    mutationFn: () => createAdminActor({ user_id: userId.trim(), email: email.trim(), display_name: displayName.trim(), role }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "actors"] });
      setUserId("");
      setEmail("");
      setDisplayName("");
    },
  });

  const handleCreate = (event: FormEvent) => {
    event.preventDefault();
    if (userId.trim() && email.trim() && displayName.trim()) {
      createMutation.mutate();
    }
  };

  return (
    <section>
      <header className="admin-page__header">
        <h1>Permissions</h1>
        <p className="admin-page__subtitle">Roles are fixed; permissions follow role membership.</p>
      </header>

      {permissionsLoading && <p className="admin-status">Loading…</p>}

      {permissions && (
        <div className="admin-section">
          <h2 className="admin-section__title">
            Your role: {permissions.current_role.replace(/_/g, " ")}
          </h2>
          <div className="admin-card">
            <p>{permissions.current_permissions.join(", ")}</p>
          </div>
          <div className="admin-section">
            <h2 className="admin-section__title">All roles</h2>
            <AdminTable
              rows={permissions.roles}
              rowKey={(row) => row.role}
              columns={[
                { key: "role", header: "Role", render: (r) => r.role.replace(/_/g, " ") },
                { key: "permissions", header: "Permissions", render: (r) => r.permissions.join(", ") },
              ]}
            />
          </div>
        </div>
      )}

      {canManage && (
        <div className="admin-section">
          <h2 className="admin-section__title">Admin actors</h2>
          <form onSubmit={handleCreate} className="admin-card admin-section admin-form-grid">
            <div className="admin-form-row">
              <label htmlFor="actor-user-id">User ID</label>
              <input
                id="actor-user-id"
                className="admin-text-input"
                placeholder="UUID of the platform user"
                value={userId}
                onChange={(event) => setUserId(event.target.value)}
                required
              />
            </div>
            <div className="admin-form-row">
              <label htmlFor="actor-email">Email</label>
              <input
                id="actor-email"
                type="email"
                className="admin-text-input"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                required
              />
            </div>
            <div className="admin-form-row">
              <label htmlFor="actor-name">Display name</label>
              <input
                id="actor-name"
                className="admin-text-input"
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
                required
              />
            </div>
            <div className="admin-form-row">
              <label htmlFor="actor-role">Role</label>
              <select
                id="actor-role"
                className="admin-select"
                value={role}
                onChange={(event) => setRole(event.target.value as AdminRole)}
              >
                {ROLES.map((r) => (
                  <option key={r} value={r}>
                    {r.replace(/_/g, " ")}
                  </option>
                ))}
              </select>
            </div>
            <button type="submit" className="admin-button" disabled={createMutation.isPending}>
              {createMutation.isPending ? "Creating…" : "Create actor"}
            </button>
          </form>

          {actorsLoading && <p className="admin-status">Loading actors…</p>}
          {actors && (
            <AdminTable
              rows={actors.items}
              rowKey={(row) => row.id}
              emptyMessage="No admin actors yet."
              columns={[
                { key: "email", header: "Email", render: (r) => r.email },
                { key: "name", header: "Name", render: (r) => r.display_name },
                { key: "active", header: "Active", render: (r) => (r.is_active ? "Yes" : "No") },
                { key: "role", header: "Role", render: (r) => <ActorRoleAction actor={r} /> },
              ]}
            />
          )}
        </div>
      )}
    </section>
  );
}
