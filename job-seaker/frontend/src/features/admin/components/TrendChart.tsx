import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { PlatformTrendPoint } from "@/features/admin/types";

interface TrendChartProps {
  data: PlatformTrendPoint[];
  color?: string;
  height?: number;
}

export function TrendChart({ data, color = "var(--adm-blue)", height = 160 }: TrendChartProps) {
  if (data.length === 0) {
    return <p className="admin-empty">No data for this period.</p>;
  }
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid stroke="var(--adm-gridline)" vertical={false} strokeWidth={1} />
        <XAxis
          dataKey="bucket"
          tick={{ fill: "var(--adm-ink-muted)", fontSize: 11 }}
          axisLine={{ stroke: "var(--adm-gridline)" }}
          tickLine={false}
          minTickGap={24}
        />
        <YAxis
          tick={{ fill: "var(--adm-ink-muted)", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={32}
          allowDecimals={false}
        />
        <Tooltip
          contentStyle={{
            background: "var(--adm-surface)",
            border: "1px solid var(--adm-border)",
            borderRadius: 8,
            fontSize: 12,
          }}
        />
        <Line
          type="monotone"
          dataKey="count"
          stroke={color}
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
