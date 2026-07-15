import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";

import { createFeatureFlag, deleteFeatureFlag, listFeatureFlags, updateFeatureFlag } from "@/features/admin/api";
import { AdminTable } from "@/features/admin/components/AdminTable";
import { useHasPermission } from "@/features/admin/context";
import type { FeatureFlagResponse } from "@/features/admin/types";

function FlagRowActions({ flag, canManage }: { flag: FeatureFlagResponse; canManage: boolean }) {
  const queryClient = useQueryClient();
  const [rollout, setRollout] = useState(flag.rollout_percentage);

  const toggleMutation = useMutation({
    mutationFn: () => updateFeatureFlag(flag.key, { is_enabled: !flag.is_enabled }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "feature-flags"] }),
  });
  const rolloutMutation = useMutation({
    mutationFn: () => updateFeatureFlag(flag.key, { rollout_percentage: rollout }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "feature-flags"] }),
  });
  const deleteMutation = useMutation({
    mutationFn: () => deleteFeatureFlag(flag.key),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "feature-flags"] }),
  });

  if (!canManage) {
    return <span>{flag.is_enabled ? "Enabled" : "Disabled"}</span>;
  }

  return (
    <div className="admin-inline-form">
      <button type="button" className="admin-button" disabled={toggleMutation.isPending} onClick={() => toggleMutation.mutate()}>
        {flag.is_enabled ? "Disable" : "Enable"}
      </button>
      <input
        type="number"
        min={0}
        max={100}
        className="admin-text-input"
        style={{ width: "4.5rem" }}
        aria-label={`Rollout percentage for ${flag.key}`}
        value={rollout}
        onChange={(event) => setRollout(Number(event.target.value))}
      />
      <button type="button" className="admin-button admin-button--ghost" disabled={rolloutMutation.isPending} onClick={() => rolloutMutation.mutate()}>
        Save %
      </button>
      <button
        type="button"
        className="admin-button admin-button--danger"
        disabled={deleteMutation.isPending}
        onClick={() => deleteMutation.mutate()}
      >
        Delete
      </button>
    </div>
  );
}

export function FeatureFlagsPage() {
  const canManage = useHasPermission("feature_flags.manage");
  const queryClient = useQueryClient();
  const [key, setKey] = useState("");
  const [name, setName] = useState("");

  const { data, isLoading, isError } = useQuery({
    queryKey: ["admin", "feature-flags"],
    queryFn: listFeatureFlags,
  });

  const createMutation = useMutation({
    mutationFn: () => createFeatureFlag({ key: key.trim(), name: name.trim() }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin", "feature-flags"] });
      setKey("");
      setName("");
    },
  });

  const handleCreate = (event: FormEvent) => {
    event.preventDefault();
    if (key.trim() && name.trim()) {
      createMutation.mutate();
    }
  };

  return (
    <section>
      <header className="admin-page__header">
        <h1>Feature Flags</h1>
        <p className="admin-page__subtitle">Toggle platform features and control rollout percentage.</p>
      </header>

      {canManage && (
        <form onSubmit={handleCreate} className="admin-card admin-section admin-form-grid">
          <div className="admin-form-row">
            <label htmlFor="flag-key">Key</label>
            <input
              id="flag-key"
              className="admin-text-input"
              value={key}
              onChange={(event) => setKey(event.target.value)}
              required
            />
          </div>
          <div className="admin-form-row">
            <label htmlFor="flag-name">Name</label>
            <input
              id="flag-name"
              className="admin-text-input"
              value={name}
              onChange={(event) => setName(event.target.value)}
              required
            />
          </div>
          <button type="submit" className="admin-button" disabled={createMutation.isPending}>
            {createMutation.isPending ? "Creating…" : "Create flag"}
          </button>
        </form>
      )}

      {isLoading && <p className="admin-status">Loading…</p>}
      {isError && <p className="admin-status admin-status--error">Unable to load feature flags.</p>}

      {data && (
        <AdminTable
          rows={data}
          rowKey={(row) => row.key}
          emptyMessage="No feature flags yet."
          columns={[
            { key: "key", header: "Key", render: (r) => <span className="admin-mono">{r.key}</span> },
            { key: "name", header: "Name", render: (r) => r.name },
            { key: "rollout", header: "Rollout", align: "right", render: (r) => `${r.rollout_percentage}%` },
            { key: "actions", header: "Status", render: (r) => <FlagRowActions flag={r} canManage={canManage} /> },
          ]}
        />
      )}
    </section>
  );
}
