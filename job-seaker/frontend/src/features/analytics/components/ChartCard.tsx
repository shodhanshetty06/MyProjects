import { motion } from "framer-motion";
import type { ReactNode } from "react";

import { fadeInUp } from "@/components/motion/variants";
import { cn } from "@/lib/cn";

interface ChartCardProps {
  title: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}

export function ChartCard({ title, subtitle, children, className }: ChartCardProps) {
  return (
    <motion.section
      variants={fadeInUp}
      className={cn("rounded-[var(--radius-lg)] border border-border bg-surface p-5", className)}
    >
      <header className="mb-3">
        <h2 className="text-sm font-semibold text-ink">{title}</h2>
        {subtitle && <p className="text-xs text-ink-secondary">{subtitle}</p>}
      </header>
      <div>{children}</div>
    </motion.section>
  );
}
