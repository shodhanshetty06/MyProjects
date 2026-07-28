import { Check, Loader2, X } from "lucide-react";
import { useEffect } from "react";

import { Button } from "@/components/ui/Button";
import { ProgressBar } from "@/components/ui/ProgressBar";
import type { PipelineStep } from "@/features/workspace/types";
import { usePipelineProgress } from "@/features/workspace/hooks";
import { cn } from "@/lib/cn";

const STEP_LABELS: Record<string, string> = {
  parsing: "Extracting your resume",
  embedding: "Understanding your experience",
  scoring: "Scoring resume quality",
  ats_baseline: "Checking ATS readiness",
  matching: "Matching you to jobs",
  recommendations: "Building recommendations",
  skill_gap: "Analyzing skill gaps",
  learning_recommendations: "Curating learning resources",
  career_health: "Scoring career health",
  dashboard: "Preparing your workspace",
  notification: "Wrapping up",
};

const PLACEHOLDER_STEPS: PipelineStep[] = Object.keys(STEP_LABELS).map((name) => ({
  name,
  status: "pending",
  started_at: null,
  completed_at: null,
  error: null,
}));

interface ProcessingStepProps {
  userId: string;
  onComplete: () => void;
}

export function ProcessingStep({ userId, onComplete }: ProcessingStepProps) {
  const { data: status, isLoading } = usePipelineProgress(userId);

  useEffect(() => {
    if (status?.status === "completed") {
      onComplete();
    }
  }, [status?.status, onComplete]);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-ink">Setting up your workspace</h2>
        <p className="mt-1 text-sm text-ink-secondary">
          This usually takes under a minute — parsing, scoring, and matching all run automatically.
        </p>
      </div>

      <ProgressBar value={status?.progress_pct ?? (isLoading ? 5 : 0)} />

      <ul className="space-y-2.5">
        {(status?.steps ?? PLACEHOLDER_STEPS).map(
          (step) => (
            <li key={step.name} className="flex items-center gap-3 text-sm">
              <span
                className={cn(
                  "flex size-5 shrink-0 items-center justify-center rounded-full",
                  step.status === "completed" && "bg-success-soft text-success",
                  step.status === "running" && "bg-brand-soft text-brand",
                  step.status === "failed" && "bg-danger-soft text-danger",
                  step.status === "pending" && "bg-surface-2 text-ink-muted",
                )}
              >
                {step.status === "completed" && <Check className="size-3" aria-hidden />}
                {step.status === "running" && <Loader2 className="size-3 animate-spin" aria-hidden />}
                {step.status === "failed" && <X className="size-3" aria-hidden />}
              </span>
              <span
                className={cn(
                  "text-ink-secondary",
                  (step.status === "running" || step.status === "completed") && "font-medium text-ink",
                )}
              >
                {STEP_LABELS[step.name] ?? step.name}
              </span>
            </li>
          ),
        )}
      </ul>

      {status?.status === "failed" && (
        <div className="space-y-3">
          <p className="text-sm text-danger">
            Something went wrong finishing setup — your workspace may be missing a few sections, but
            you can continue and retry from there.
          </p>
          <Button variant="secondary" onClick={onComplete}>
            Continue to workspace
          </Button>
        </div>
      )}
    </div>
  );
}
