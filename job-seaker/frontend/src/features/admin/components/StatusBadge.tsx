const TONE_BY_STATUS: Record<string, "neutral" | "good" | "warning" | "bad"> = {
  active: "good",
  verified: "good",
  resolved: "good",
  success: "good",
  pending: "neutral",
  pending_review: "warning",
  unverified: "neutral",
  reviewing: "neutral",
  open: "warning",
  flagged: "warning",
  warning: "warning",
  suspended: "warning",
  rejected: "bad",
  removed: "bad",
  banned: "bad",
  dismissed: "neutral",
  error: "bad",
  failed: "bad",
};

interface StatusBadgeProps {
  status: string;
}

export function StatusBadge({ status }: StatusBadgeProps) {
  const tone = TONE_BY_STATUS[status] ?? "neutral";
  return <span className={`admin-badge admin-badge--${tone}`}>{status.replace(/_/g, " ")}</span>;
}
