import type { HTMLAttributes } from "react";

import { cn } from "@/lib/cn";

export type BadgeTone = "neutral" | "brand" | "success" | "warning" | "danger" | "info";

const TONE_CLASSES: Record<BadgeTone, string> = {
  neutral: "bg-surface-2 text-ink-secondary border-border",
  brand: "bg-brand-soft text-brand border-brand-border",
  success: "bg-success-soft text-success border-transparent",
  warning: "bg-warning-soft text-warning border-transparent",
  danger: "bg-danger-soft text-danger border-transparent",
  info: "bg-info-soft text-info border-transparent",
};

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: BadgeTone;
}

export function Badge({ tone = "neutral", className, ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium leading-4",
        TONE_CLASSES[tone],
        className,
      )}
      {...props}
    />
  );
}
