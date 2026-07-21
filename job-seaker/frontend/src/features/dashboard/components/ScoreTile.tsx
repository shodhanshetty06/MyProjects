import { motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { fadeInUp } from "@/components/motion/variants";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { ScoreRing } from "@/components/ui/ScoreRing";
import { Skeleton } from "@/components/ui/Skeleton";
import { cn } from "@/lib/cn";

interface ScoreTileProps {
  icon: LucideIcon;
  label: string;
  hint?: string;
  isLoading?: boolean;
  value?: number | null;
  emptyLabel?: string;
  emptyAction?: ReactNode;
  onCompute?: () => void;
  computeLabel?: string;
  isComputing?: boolean;
  className?: string;
}

export function ScoreTile({
  icon: Icon,
  label,
  hint,
  isLoading,
  value,
  emptyLabel,
  emptyAction,
  onCompute,
  computeLabel = "Calculate",
  isComputing,
  className,
}: ScoreTileProps) {
  return (
    <motion.div variants={fadeInUp}>
      <Card className={cn("flex items-center gap-4 p-4", className)}>
        {isLoading ? (
          <Skeleton className="size-[52px] shrink-0 rounded-full" />
        ) : typeof value === "number" ? (
          <ScoreRing value={value} size={52} strokeWidth={5} />
        ) : (
          <div className="flex size-[52px] shrink-0 items-center justify-center rounded-full bg-surface-2 text-ink-muted">
            <Icon className="size-5" aria-hidden />
          </div>
        )}

        <div className="min-w-0 flex-1">
          <p className="flex items-center gap-1.5 text-xs font-medium text-ink-secondary">
            <Icon className="size-3.5 text-ink-muted" aria-hidden />
            {label}
          </p>
          {isLoading ? (
            <Skeleton className="mt-1.5 h-3 w-3/4" />
          ) : typeof value === "number" ? (
            <p className="mt-0.5 truncate text-xs text-ink-muted">{hint}</p>
          ) : onCompute ? (
            <Button size="sm" variant="ghost" className="-ml-2 mt-0.5 h-6 px-2 text-xs" onClick={onCompute} isLoading={isComputing}>
              {computeLabel}
            </Button>
          ) : (
            <p className="mt-0.5 truncate text-xs text-ink-muted">{emptyLabel}</p>
          )}
          {emptyAction}
        </div>
      </Card>
    </motion.div>
  );
}
