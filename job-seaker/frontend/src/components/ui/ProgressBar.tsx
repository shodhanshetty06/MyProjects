import { motion } from "framer-motion";

import { cn } from "@/lib/cn";

interface ProgressBarProps {
  value: number;
  max?: number;
  tone?: "brand" | "success" | "warning" | "danger";
  className?: string;
  trackClassName?: string;
}

const TONE_CLASSES = {
  brand: "bg-brand",
  success: "bg-success",
  warning: "bg-warning",
  danger: "bg-danger",
};

export function ProgressBar({ value, max = 100, tone = "brand", className, trackClassName }: ProgressBarProps) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div
      role="progressbar"
      aria-valuenow={value}
      aria-valuemin={0}
      aria-valuemax={max}
      className={cn("h-1.5 w-full overflow-hidden rounded-full bg-surface-2", trackClassName, className)}
    >
      <motion.div
        className={cn("h-full rounded-full", TONE_CLASSES[tone])}
        initial={{ width: 0 }}
        animate={{ width: `${pct}%` }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      />
    </div>
  );
}
