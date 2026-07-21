import { motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";

import { fadeInUp } from "@/components/motion/variants";
import { cn } from "@/lib/cn";

interface FeatureCardProps {
  icon: LucideIcon;
  title: string;
  description: string;
  className?: string;
}

export function FeatureCard({ icon: Icon, title, description, className }: FeatureCardProps) {
  return (
    <motion.div
      variants={fadeInUp}
      className={cn(
        "group rounded-[var(--radius-xl)] border border-border bg-surface p-6 transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md",
        className,
      )}
    >
      <div className="mb-4 flex size-10 items-center justify-center rounded-[var(--radius-md)] bg-brand-soft text-brand transition-colors group-hover:bg-brand group-hover:text-brand-ink">
        <Icon className="size-5" aria-hidden />
      </div>
      <h3 className="text-sm font-semibold text-ink">{title}</h3>
      <p className="mt-1.5 text-sm leading-relaxed text-ink-secondary">{description}</p>
    </motion.div>
  );
}
