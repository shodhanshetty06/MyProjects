import { Badge } from "@/components/ui/Badge";

export type ReviewFieldState = "ai" | "accepted" | "added" | "missing";

const BADGE_CONFIG: Record<ReviewFieldState, { label: string; tone: "info" | "success" | "neutral" | "warning" }> = {
  ai: { label: "AI-filled", tone: "info" },
  accepted: { label: "Accepted", tone: "success" },
  added: { label: "Added by you", tone: "neutral" },
  missing: { label: "Missing", tone: "warning" },
};

/** Marks whether a field/item came from the AI extraction, was accepted,
 * was added manually, or is still missing — the highlighting the Profile
 * Review screen uses throughout. */
export function ReviewBadge({ state }: { state: ReviewFieldState }) {
  const config = BADGE_CONFIG[state];
  return <Badge tone={config.tone}>{config.label}</Badge>;
}
