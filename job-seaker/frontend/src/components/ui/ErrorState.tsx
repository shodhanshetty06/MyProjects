import { AlertTriangle } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/cn";

interface ErrorStateProps {
  title?: string;
  description?: string;
  action?: ReactNode;
  className?: string;
}

export function ErrorState({
  title = "Something went wrong",
  description = "That request failed. Try again in a moment.",
  action,
  className,
}: ErrorStateProps) {
  return (
    <div
      role="alert"
      className={cn(
        "flex flex-col items-center justify-center gap-3 rounded-[var(--radius-lg)] border border-danger-soft bg-danger-soft/40 px-6 py-12 text-center",
        className,
      )}
    >
      <div className="flex size-11 items-center justify-center rounded-full bg-danger-soft text-danger">
        <AlertTriangle className="size-5" aria-hidden />
      </div>
      <div className="space-y-1">
        <p className="text-sm font-medium text-ink">{title}</p>
        <p className="max-w-sm text-sm text-ink-secondary">{description}</p>
      </div>
      {action}
    </div>
  );
}
