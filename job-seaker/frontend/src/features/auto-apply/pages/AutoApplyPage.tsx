import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Send } from "lucide-react";
import { Link } from "react-router-dom";

import { PageHeader } from "@/components/layout/PageHeader";
import { staggerChildren } from "@/components/motion/variants";
import { Card, CardContent } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { ErrorState } from "@/components/ui/ErrorState";
import { SkeletonCard } from "@/components/ui/Skeleton";
import { useAuth } from "@/features/auth";
import { listAutoApplyTasks } from "@/features/auto-apply/api";
import { TaskStatusBadge } from "@/features/auto-apply/components/TaskStatusBadge";
import { formatRelativeDate } from "@/lib/format";
import { paths } from "@/routes/paths";

export function AutoApplyPage() {
  const { user } = useAuth();
  const userId = user!.id;

  const tasksQuery = useQuery({
    queryKey: ["auto-apply-tasks", userId],
    queryFn: () => listAutoApplyTasks(userId, { page_size: 50 }),
  });

  const tasks = tasksQuery.data?.data ?? [];

  return (
    <div>
      <PageHeader
        eyebrow="Applications"
        title="Auto apply"
        description="Applications queued and submitted automatically, with full attempt history."
      />

      {tasksQuery.isLoading ? (
        <div className="grid gap-3 sm:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      ) : tasksQuery.isError ? (
        <ErrorState description="Couldn't load auto-apply tasks." />
      ) : tasks.length === 0 ? (
        <EmptyState
          icon={Send}
          title="No auto-apply tasks yet"
          description="Queue auto-apply from a job's detail page to see it tracked here."
          action={
            <Link to={paths.search} className="text-sm font-medium text-brand hover:underline">
              Browse jobs
            </Link>
          }
        />
      ) : (
        <motion.div variants={staggerChildren(0.04)} initial="hidden" animate="visible" className="grid gap-3 sm:grid-cols-2">
          {tasks.map((task) => (
            <motion.div key={task.id} variants={{ hidden: { opacity: 0, y: 6 }, visible: { opacity: 1, y: 0 } }}>
              <Link to={paths.autoApplyTask(task.id)}>
                <Card className="transition-all hover:-translate-y-0.5 hover:border-brand-border hover:shadow-sm">
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium text-ink">{task.job_title}</p>
                        <p className="truncate text-xs text-ink-secondary">{task.company_name}</p>
                      </div>
                      <TaskStatusBadge status={task.status} />
                    </div>
                    <div className="mt-2 flex items-center justify-between text-xs text-ink-muted">
                      <span>
                        Attempt {task.attempt_count}/{task.max_attempts}
                      </span>
                      <span>{formatRelativeDate(task.created_at)}</span>
                    </div>
                  </CardContent>
                </Card>
              </Link>
            </motion.div>
          ))}
        </motion.div>
      )}
    </div>
  );
}
