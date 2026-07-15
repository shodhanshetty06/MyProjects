interface StatTileProps {
  label: string;
  value: string;
  hint?: string;
}

export function StatTile({ label, value, hint }: StatTileProps) {
  return (
    <div className="analytics-stat-tile">
      <p className="analytics-stat-tile__label">{label}</p>
      <p className="analytics-stat-tile__value">{value}</p>
      {hint && <p className="analytics-stat-tile__hint">{hint}</p>}
    </div>
  );
}
