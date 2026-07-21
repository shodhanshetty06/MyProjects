import { motion } from "framer-motion";
import { Minus, Plus, Target } from "lucide-react";

import { fadeInUp } from "@/components/motion/variants";
import { Card, CardContent } from "@/components/ui/Card";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { Skeleton } from "@/components/ui/Skeleton";
import { useGoalsStore } from "@/stores/goalsStore";

interface WeeklyGoalCardProps {
  applicationsThisWeek?: number;
  isLoading?: boolean;
}

export function WeeklyGoalCard({ applicationsThisWeek, isLoading }: WeeklyGoalCardProps) {
  const goal = useGoalsStore((state) => state.weeklyApplicationGoal);
  const setGoal = useGoalsStore((state) => state.setWeeklyApplicationGoal);
  const applied = applicationsThisWeek ?? 0;
  const met = applied >= goal;

  return (
    <motion.div variants={fadeInUp}>
      <Card>
        <CardContent className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex size-10 shrink-0 items-center justify-center rounded-[var(--radius-md)] bg-brand-soft text-brand">
              <Target className="size-5" aria-hidden />
            </div>
            <div>
              <p className="text-sm font-medium text-ink">Weekly application goal</p>
              {isLoading ? (
                <Skeleton className="mt-1 h-3 w-32" />
              ) : (
                <p className="text-xs text-ink-secondary">
                  {applied} of {goal} applications this week
                  {met && " · goal met"}
                </p>
              )}
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="w-40">
              <ProgressBar value={applied} max={goal} tone={met ? "success" : "brand"} />
            </div>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => setGoal(goal - 1)}
                className="flex size-6 items-center justify-center rounded-[var(--radius-sm)] text-ink-muted transition-colors hover:bg-surface-2 hover:text-ink"
                aria-label="Decrease weekly goal"
              >
                <Minus className="size-3.5" aria-hidden />
              </button>
              <span className="w-4 text-center text-xs font-medium text-ink">{goal}</span>
              <button
                type="button"
                onClick={() => setGoal(goal + 1)}
                className="flex size-6 items-center justify-center rounded-[var(--radius-sm)] text-ink-muted transition-colors hover:bg-surface-2 hover:text-ink"
                aria-label="Increase weekly goal"
              >
                <Plus className="size-3.5" aria-hidden />
              </button>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
