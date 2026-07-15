import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { chrome, pick, useColorScheme } from "@/features/analytics/theme";

export interface ColumnDatum {
  label: string;
  value: number;
}

interface ColumnChartProps {
  data: ColumnDatum[];
  color: string;
  height?: number;
}

interface TooltipPayloadEntry {
  value?: number;
  payload?: ColumnDatum;
}

function ColumnTooltip({
  active,
  payload,
  scheme,
}: {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
  scheme: "light" | "dark";
}) {
  if (!active || !payload || payload.length === 0) {
    return null;
  }
  const datum = payload[0].payload;
  if (!datum) {
    return null;
  }
  return (
    <div
      className="analytics-tooltip"
      style={{
        background: pick(chrome.surface, scheme),
        color: pick(chrome.primaryInk, scheme),
        borderColor: pick(chrome.gridline, scheme),
      }}
    >
      <p className="analytics-tooltip__label">{datum.label}</p>
      <p className="analytics-tooltip__row">
        <span className="analytics-tooltip__value">{datum.value}</span>
      </p>
    </div>
  );
}

export function ColumnChart({ data, color, height = 200 }: ColumnChartProps) {
  const scheme = useColorScheme();
  const gridline = pick(chrome.gridline, scheme);
  const muted = pick(chrome.mutedInk, scheme);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={gridline} vertical={false} strokeWidth={1} />
        <XAxis
          dataKey="label"
          tick={{ fill: muted, fontSize: 12 }}
          axisLine={{ stroke: gridline }}
          tickLine={false}
        />
        <YAxis tick={{ fill: muted, fontSize: 12 }} axisLine={false} tickLine={false} width={32} />
        <Tooltip content={<ColumnTooltip scheme={scheme} />} cursor={{ fill: gridline, opacity: 0.4 }} />
        <Bar dataKey="value" fill={color} radius={[4, 4, 0, 0]} maxBarSize={24} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}
