interface StatTileProps {
  label: string;
  value: string;
  hint?: string;
}

export function StatTile({ label, value, hint }: StatTileProps) {
  return (
    <div className="rounded-[var(--radius-md)] border border-border bg-surface p-3.5">
      <p className="text-xs text-ink-secondary">{label}</p>
      <p className="mt-1 text-lg font-semibold text-ink">{value}</p>
      {hint && <p className="mt-0.5 text-[11px] text-ink-muted">{hint}</p>}
    </div>
  );
}
