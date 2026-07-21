export function formatSalaryRange(
  min: number | null | undefined,
  max: number | null | undefined,
  currency?: string | null,
  period?: string | null,
): string | null {
  if (!min && !max) return null;
  const cur = currency ?? "USD";
  const fmt = (n: number) =>
    new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: cur,
      maximumFractionDigits: 0,
      notation: n >= 100_000 ? "compact" : "standard",
    }).format(n);

  const range = min && max ? `${fmt(min)} - ${fmt(max)}` : fmt((min ?? max) as number);
  return period ? `${range} / ${period}` : range;
}

export function formatLocation(location: { city?: string | null; region?: string | null; country?: string | null } | null): string {
  if (!location) return "Location not specified";
  return [location.city, location.region, location.country].filter(Boolean).join(", ") || "Location not specified";
}

export function formatRelativeDate(iso: string | null | undefined): string {
  if (!iso) return "Recently";
  const diffMs = Date.now() - new Date(iso).getTime();
  const minutes = Math.floor(diffMs / 60_000);
  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  const weeks = Math.floor(days / 7);
  if (weeks < 5) return `${weeks}w ago`;
  const months = Math.floor(days / 30);
  if (months < 12) return `${months}mo ago`;
  return new Date(iso).toLocaleDateString();
}

export function formatEnumLabel(value: string | null | undefined): string {
  if (!value) return "";
  return value
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
