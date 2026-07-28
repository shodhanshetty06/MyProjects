import { Loader2 } from "lucide-react";

import { Card } from "@/components/ui/Card";
import { ProgressBar } from "@/components/ui/ProgressBar";
import type { PipelineStatus } from "@/features/workspace/types";

const STEP_LABELS: Record<string, string> = {
  parsing: "Extracting your resume",
  embedding: "Understanding your experience",
  scoring: "Scoring resume quality",
  ats_baseline: "Checking ATS readiness",
  matching: "Matching you to jobs",
  recommendations: "Building recommendations",
  dashboard: "Preparing your workspace",
  notification: "Wrapping up",
};

/** Slim banner shown while a resume update's pipeline is still running —
 * lets the user see live progress without leaving the workspace. */
export function PipelineBanner({ status }: { status: PipelineStatus }) {
  return (
    <Card className="mb-6 flex items-center gap-4 p-4">
      <Loader2 className="size-5 shrink-0 animate-spin text-brand" aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-ink">
          {status.current_step ? STEP_LABELS[status.current_step] ?? status.current_step : "Processing"}
        </p>
        <ProgressBar value={status.progress_pct} className="mt-1.5" />
      </div>
      <span className="shrink-0 text-xs font-medium text-ink-muted">{status.progress_pct}%</span>
    </Card>
  );
}
