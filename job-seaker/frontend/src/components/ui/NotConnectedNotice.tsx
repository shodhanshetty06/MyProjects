import { motion } from "framer-motion";
import { PlugZap } from "lucide-react";

import { fadeIn } from "@/components/motion/variants";
import { cn } from "@/lib/cn";

interface NotConnectedNoticeProps {
  title?: string;
  description: string;
  className?: string;
}

/**
 * Consistent, honest empty-state for screens whose form/UI is fully built
 * but has no backend to submit to yet (no fake success, no silent no-op).
 * Used by Contact, Forgot/Reset Password, Email Verification, Security
 * password change, and API Keys.
 */
export function NotConnectedNotice({ title = "Not connected yet", description, className }: NotConnectedNoticeProps) {
  return (
    <motion.div
      variants={fadeIn}
      initial="hidden"
      animate="visible"
      role="status"
      className={cn(
        "flex items-start gap-3 rounded-[var(--radius-md)] border border-warning-soft bg-warning-soft/60 p-3.5",
        className,
      )}
    >
      <PlugZap className="mt-0.5 size-4 shrink-0 text-warning" aria-hidden />
      <div>
        <p className="text-sm font-medium text-ink">{title}</p>
        <p className="mt-0.5 text-sm text-ink-secondary">{description}</p>
      </div>
    </motion.div>
  );
}
