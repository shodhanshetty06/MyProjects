import { motion } from "framer-motion";
import { CheckCircle2, Loader2, XCircle } from "lucide-react";

import { Badge, type BadgeTone } from "@/components/ui/Badge";
import type { ResumeVersionStatus } from "@/features/resume-parser/types";

const STATUS_TONE: Record<ResumeVersionStatus, BadgeTone> = {
  uploaded: "neutral",
  parsing: "info",
  parsed: "success",
  failed: "danger",
};

const STATUS_LABEL: Record<ResumeVersionStatus, string> = {
  uploaded: "Uploaded",
  parsing: "Parsing...",
  parsed: "Parsed",
  failed: "Failed",
};

export function VersionStatusBadge({ status }: { status: ResumeVersionStatus }) {
  return (
    <Badge tone={STATUS_TONE[status]}>
      {status === "parsing" ? (
        <motion.span animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: "linear" }}>
          <Loader2 className="size-3" aria-hidden />
        </motion.span>
      ) : status === "parsed" ? (
        <CheckCircle2 className="size-3" aria-hidden />
      ) : status === "failed" ? (
        <XCircle className="size-3" aria-hidden />
      ) : null}
      {STATUS_LABEL[status]}
    </Badge>
  );
}
