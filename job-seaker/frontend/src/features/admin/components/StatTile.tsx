interface StatTileProps {
  label: string;
  value: string | number;
  hint?: string;
}

export function StatTile({ label, value, hint }: StatTileProps) {
  return (
    <div className="admin-stat-tile">
      <p className="admin-stat-tile__label">{label}</p>
      <p className="admin-stat-tile__value">{value}</p>
      {hint && <p className="admin-stat-tile__hint">{hint}</p>}
    </div>
  );
}
