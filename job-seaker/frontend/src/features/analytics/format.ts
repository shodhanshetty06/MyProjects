import type { TrendInterval } from "@/features/analytics/types";

export function formatCompactNumber(value: number): string {
  return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(
    value,
  );
}

export function formatPercent(value: number): string {
  return new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 0 }).format(
    value,
  );
}

export function formatDays(value: number | null): string {
  if (value === null) {
    return "N/A";
  }
  return `${value.toFixed(1)}d`;
}

export function formatBucketLabel(bucket: string, interval: TrendInterval): string {
  const date = new Date(bucket);
  if (interval === "month") {
    return date.toLocaleDateString(undefined, { month: "short", year: "2-digit" });
  }
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function formatDateLabel(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}
