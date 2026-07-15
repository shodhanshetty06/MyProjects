import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { formatBucketLabel } from "@/features/analytics/format";
import { chrome, pick, useColorScheme } from "@/features/analytics/theme";
import type { TrendInterval } from "@/features/analytics/types";

export interface TrendLineSeries {
  key: string;
  label: string;
  color: string;
}

interface TrendLineChartProps<T extends { bucket: string }> {
  data: readonly T[];
  interval: TrendInterval;
  series: TrendLineSeries[];
  valueFormatter?: (value: number) => string;
  height?: number;
}

interface TooltipPayloadEntry {
  dataKey?: string;
  value?: number;
  color?: string;
}

function ChartTooltip({
  active,
  payload,
  label,
  interval,
  series,
  valueFormatter,
  scheme,
}: {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
  label?: string;
  interval: TrendInterval;
  series: TrendLineSeries[];
  valueFormatter: (value: number) => string;
  scheme: "light" | "dark";
}) {
  if (!active || !payload || payload.length === 0 || !label) {
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
      <p className="analytics-tooltip__label">{formatBucketLabel(label, interval)}</p>
      {series.map((s) => {
        const entry = payload.find((p) => p.dataKey === s.key);
        if (!entry || entry.value === undefined) {
          return null;
        }
        return (
          <p key={s.key} className="analytics-tooltip__row">
            <span className="analytics-tooltip__key" style={{ backgroundColor: s.color }} />
            <span className="analytics-tooltip__value">{valueFormatter(entry.value)}</span>
            {series.length > 1 && <span className="analytics-tooltip__series">{s.label}</span>}
          </p>
        );
      })}
    </div>
  );
}

export function TrendLineChart<T extends { bucket: string }>({
  data,
  interval,
  series,
  valueFormatter = (value) => String(value),
  height = 220,
}: TrendLineChartProps<T>) {
  const scheme = useColorScheme();
  const gridline = pick(chrome.gridline, scheme);
  const muted = pick(chrome.mutedInk, scheme);
  const surface = pick(chrome.surface, scheme);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={gridline} vertical={false} strokeWidth={1} />
        <XAxis
          dataKey="bucket"
          tickFormatter={(value: string) => formatBucketLabel(value, interval)}
          tick={{ fill: muted, fontSize: 12 }}
          axisLine={{ stroke: gridline }}
          tickLine={false}
          minTickGap={24}
        />
        <YAxis
          tick={{ fill: muted, fontSize: 12 }}
          axisLine={false}
          tickLine={false}
          width={40}
          tickFormatter={(value: number) => valueFormatter(value)}
        />
        <Tooltip
          content={
            <ChartTooltip
              interval={interval}
              series={series}
              valueFormatter={valueFormatter}
              scheme={scheme}
            />
          }
        />
        {series.length > 1 && (
          <Legend
            wrapperStyle={{ fontSize: 12, color: pick(chrome.secondaryInk, scheme) }}
            iconType="plainline"
          />
        )}
        {series.map((s) => (
          <Line
            key={s.key}
            type="monotone"
            dataKey={s.key}
            name={s.label}
            stroke={s.color}
            strokeWidth={2}
            dot={{ r: 4, fill: s.color, stroke: surface, strokeWidth: 2 }}
            activeDot={{ r: 5, fill: s.color, stroke: surface, strokeWidth: 2 }}
            connectNulls
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
