import { motion } from "framer-motion";
import type { ReactNode } from "react";

import { fadeInUp } from "@/components/motion/variants";
import { cn } from "@/lib/cn";

interface PageHeaderProps {
  eyebrow?: string;
  title: string;
  description?: string;
  actions?: ReactNode;
  className?: string;
}

export function PageHeader({ eyebrow, title, description, actions, className }: PageHeaderProps) {
  return (
    <motion.div
      variants={fadeInUp}
      initial="hidden"
      animate="visible"
      className={cn("mb-6 flex flex-wrap items-start justify-between gap-4", className)}
    >
      <div className="space-y-1">
        {eyebrow && (
          <p className="text-xs font-medium uppercase tracking-wide text-brand">{eyebrow}</p>
        )}
        <h1 className="text-xl font-semibold tracking-tight text-ink sm:text-2xl">{title}</h1>
        {description && <p className="max-w-2xl text-sm text-ink-secondary">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </motion.div>
  );
}
