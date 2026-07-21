import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { AlertTriangle, ExternalLink, RotateCcw, XCircle } from "lucide-react";
import { useParams } from "react-router-dom";

import { PageHeader } from "@/components/layout/PageHeader";
import { staggerChildren } from "@/components/motion/variants";
import { Badge, type BadgeTone } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent } from "@/components/ui/Card";
import { ErrorState } from "@/components/ui/ErrorState";
import { SkeletonText } from "@/components/ui/Skeleton";
import { useAuth } from "@/features/auth";
import { cancelAutoApplyTask, getAutoApplyTask, retryAutoApplyTask } from "@/features/auto-apply/api";
import { TaskStatusBadge } from "@/features/auto-apply/components/TaskStatusBadge";
import type { AutoApplyAttemptStatus } from "@/features/auto-apply/types";
import { formatEnumLabel, formatRelativeDate } from "@/lib/format";
import { toast } from "@/stores/toastStore";

const ATTEMPT_TONE: Record<AutoApplyAttemptStatus, BadgeTone> = {
  running: "info",
  success: "success",
  failed: "danger",
};

export function AutoApplyTaskDetailPage() {
  const { taskId } = useParams<{ taskId: string }>();
  const { user } = useAuth();
  const userId = user!.id;
  const queryClient = useQueryClient();

  const taskQuery = useQuery({
    queryKey: ["auto-apply-task", userId, taskId],
    queryFn: () => getAutoApplyTask(userId, taskId as string),
    enabled: !!taskId,
    refetchInterval: (query) => (query.state.data && ["queued", "running", "retrying"].includes(query.state.data.status) ? 4000 : false),
  });

  const invalidate = () => {
    void queryClient.invalidateQueries({ queryKey: ["auto-apply-task", userId, taskId] });
    void queryClient.invalidateQueries({ queryKey: ["auto-apply-tasks", userId] });
  };

  const retryMutation = useMutation({
    mutationFn: () => retryAutoApplyTask(userId, taskId as string),
    onSuccess: () => {
      toast.success("Retry queued");
      invalidate();
    },
  });

  const cancelMutation = useMutation({
    mutationFn: () => cancelAutoApplyTask(userId, taskId as string),
    onSuccess: () => {
      toast.success("Task cancelled");
      invalidate();
    },
  });

  if (taskQuery.isLoading) return <SkeletonText lines={6} />;
  if (taskQuery.isError || !taskQuery.data) return <ErrorState title="Task not found" />;

  const task = taskQuery.data;
  const canRetry = task.status === "failed" || task.status === "requires_review";
  const canCancel = ["pending", "queued", "running", "retrying"].includes(task.status);

  return (
    <div className="mx-auto max-w-2xl">
      <PageHeader
        eyebrow="Auto apply"
        title={task.job_title}
        description={task.company_name}
        actions={
          <div className="flex items-center gap-2">
            {canRetry && (
              <Button variant="secondary" onClick={() => retryMutation.mutate()} isLoading={retryMutation.isPending}>
                <RotateCcw className="size-4" aria-hidden />
                Retry
              </Button>
            )}
            {canCancel && (
              <Button variant="ghost" className="text-danger hover:bg-danger-soft" onClick={() => cancelMutation.mutate()} isLoading={cancelMutation.isPending}>
                <XCircle className="size-4" aria-hidden />
                Cancel
              </Button>
            )}
          </div>
        }
      />

      <Card className="mb-6">
        <CardContent className="flex flex-wrap items-center gap-4 p-5">
          <TaskStatusBadge status={task.status} />
          <span className="text-xs text-ink-muted">
            Attempt {task.attempt_count}/{task.max_attempts}
          </span>
          {task.resume_used_display_name && <span className="text-xs text-ink-muted">Resume: {task.resume_used_display_name}</span>}
          <a href={task.job_url} target="_blank" rel="noreferrer" className="ml-auto flex items-center gap-1 text-xs font-medium text-brand hover:underline">
            View posting
            <ExternalLink className="size-3" aria-hidden />
          </a>
        </CardContent>
      </Card>

      {task.missing_fields.length > 0 && (
        <Card className="mb-6 border-warning-soft bg-warning-soft/40">
          <CardContent className="flex items-start gap-3 p-4">
            <AlertTriangle className="mt-0.5 size-4 shrink-0 text-warning" aria-hidden />
            <div>
              <p className="text-sm font-medium text-ink">Needs manual input</p>
              <p className="mt-0.5 text-xs text-ink-secondary">{task.missing_fields.join(", ")}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {task.error_message && (
        <Card className="mb-6 border-danger-soft bg-danger-soft/40">
          <CardContent className="p-4 text-sm text-danger">{task.error_message}</CardContent>
        </Card>
      )}

      <h2 className="mb-3 text-sm font-semibold text-ink">Attempts</h2>
      {task.attempts.length === 0 ? (
        <p className="text-sm text-ink-muted">No attempts yet.</p>
      ) : (
        <motion.div variants={staggerChildren(0.05)} initial="hidden" animate="visible" className="space-y-2">
          {task.attempts.map((attempt) => (
            <motion.div key={attempt.id} variants={{ hidden: { opacity: 0, x: -8 }, visible: { opacity: 1, x: 0 } }}>
              <Card>
                <CardContent className="flex items-center gap-3 p-4">
                  <span className="text-xs font-medium text-ink-muted">#{attempt.attempt_number}</span>
                  <Badge tone={ATTEMPT_TONE[attempt.status]}>{formatEnumLabel(attempt.status)}</Badge>
                  {attempt.failure_category && <Badge tone="neutral">{formatEnumLabel(attempt.failure_category)}</Badge>}
                  <span className="ml-auto text-xs text-ink-muted">{formatRelativeDate(attempt.started_at)}</span>
                </CardContent>
                {attempt.error_message && (
                  <CardContent className="pt-0 text-xs text-danger">{attempt.error_message}</CardContent>
                )}
              </Card>
            </motion.div>
          ))}
        </motion.div>
      )}
    </div>
  );
}
