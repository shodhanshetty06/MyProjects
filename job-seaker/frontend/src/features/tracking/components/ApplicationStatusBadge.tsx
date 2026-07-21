import { Badge, type BadgeTone } from "@/components/ui/Badge";
import type { ApplicationStatus } from "@/features/tracking/types";
import { formatEnumLabel } from "@/lib/format";

export const STATUS_TONE: Record<ApplicationStatus, BadgeTone> = {
  applied: "info",
  assessment: "warning",
  interview: "brand",
  offer: "success",
  accepted: "success",
  rejected: "danger",
  withdrawn: "neutral",
};

export function ApplicationStatusBadge({ status }: { status: ApplicationStatus }) {
  return <Badge tone={STATUS_TONE[status]}>{formatEnumLabel(status)}</Badge>;
}
