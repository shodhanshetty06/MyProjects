export interface BarListItem {
  label: string;
  value: number;
  badge?: string;
}

interface BarListChartProps {
  items: BarListItem[];
  color: string;
  valueFormatter?: (value: number) => string;
}

/**
 * A horizontal bar list built from plain HTML rather than an SVG chart —
 * the direct label (value always visible at the bar's end) makes a hover
 * tooltip unnecessary, and it handles long category names (skills, company
 * names) without axis-label truncation.
 */
export function BarListChart({ items, color, valueFormatter = String }: BarListChartProps) {
  if (items.length === 0) {
    return <p className="analytics-empty">No data for this period.</p>;
  }
  const max = Math.max(1, ...items.map((item) => item.value));

  return (
    <div className="analytics-barlist">
      {items.map((item) => (
        <div className="analytics-barlist__row" key={item.label}>
          <span className="analytics-barlist__label">{item.label}</span>
          <div className="analytics-barlist__track">
            <div
              className="analytics-barlist__fill"
              style={{ width: `${(item.value / max) * 100}%`, backgroundColor: color }}
            />
          </div>
          <span className="analytics-barlist__value">
            {valueFormatter(item.value)}
            {item.badge && <span className="analytics-barlist__badge">{item.badge}</span>}
          </span>
        </div>
      ))}
    </div>
  );
}
