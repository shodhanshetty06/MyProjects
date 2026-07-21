import { motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { fadeInUp } from "@/components/motion/variants";
import { cn } from "@/lib/cn";

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: ReactNode;
  className?: string;
}

export function EmptyState({ icon: Icon, title, description, action, className }: EmptyStateProps) {
  return (
    <motion.div
      variants={fadeInUp}
      initial="hidden"
      animate="visible"
      className={cn(
        "flex flex-col items-center justify-center gap-3 rounded-[var(--radius-lg)] border border-dashed border-border px-6 py-12 text-center",
        className,
      )}
    >
      {Icon && (
        <div className="flex size-11 items-center justify-center rounded-full bg-surface-2 text-ink-muted">
          <Icon className="size-5" aria-hidden />
        </div>
      )}
      <div className="space-y-1">
        <p className="text-sm font-medium text-ink">{title}</p>
        {description && <p className="max-w-sm text-sm text-ink-secondary">{description}</p>}
      </div>
      {action}
    </motion.div>
  );
}
