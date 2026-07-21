import { Badge, type BadgeTone } from "@/components/ui/Badge";
import type { AutoApplyTaskStatus } from "@/features/auto-apply/types";
import { formatEnumLabel } from "@/lib/format";

export const TASK_STATUS_TONE: Record<AutoApplyTaskStatus, BadgeTone> = {
  pending: "neutral",
  queued: "info",
  running: "brand",
  submitted: "success",
  requires_review: "warning",
  retrying: "warning",
  failed: "danger",
  cancelled: "neutral",
};

export function TaskStatusBadge({ status }: { status: AutoApplyTaskStatus }) {
  return <Badge tone={TASK_STATUS_TONE[status]}>{formatEnumLabel(status)}</Badge>;
}
