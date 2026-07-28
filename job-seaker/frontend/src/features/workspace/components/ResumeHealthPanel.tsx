import { FileText } from "lucide-react";

import { Badge } from "@/components/ui/Badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { ScoreRing } from "@/components/ui/ScoreRing";
import { SkeletonText } from "@/components/ui/Skeleton";
import { FixSkillGapsDialog } from "@/features/workspace/components/FixSkillGapsDialog";
import { GenerateAtsResumeDialog } from "@/features/workspace/components/GenerateAtsResumeDialog";
import { ImproveResumeDialog } from "@/features/workspace/components/ImproveResumeDialog";
import { UploadResumeButton } from "@/features/workspace/components/UploadResumeButton";
import type { ResumeHealth } from "@/features/workspace/types";

interface ResumeHealthPanelProps {
  health: ResumeHealth;
  isLoading: boolean;
}

export function ResumeHealthPanel({ health, isLoading }: ResumeHealthPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-1.5">
          <FileText className="size-4 text-brand" aria-hidden />
          Resume health
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <SkeletonText lines={3} />
        ) : health.resume_version_id === null ? (
          <EmptyState
            icon={FileText}
            title="No resume yet"
            description="Upload a resume to see your quality and ATS scores here."
            action={<UploadResumeButton />}
          />
        ) : (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-2">
              <ImproveResumeDialog resumeVersionId={health.resume_version_id} />
              <GenerateAtsResumeDialog resumeVersionId={health.resume_version_id} />
            </div>
            <div className="flex items-center gap-6">
              <div className="flex flex-col items-center gap-1.5">
                {typeof health.resume_score === "number" ? (
                  <ScoreRing value={health.resume_score} size={64} strokeWidth={6} />
                ) : (
                  <div className="flex size-16 items-center justify-center rounded-full bg-surface-2 text-xs text-ink-muted">
                    ...
                  </div>
                )}
                <p className="text-xs text-ink-secondary">Resume score</p>
              </div>
              <div className="flex flex-col items-center gap-1.5">
                {typeof health.ats_baseline_score === "number" ? (
                  <ScoreRing value={health.ats_baseline_score} size={64} strokeWidth={6} />
                ) : (
                  <div className="flex size-16 items-center justify-center rounded-full bg-surface-2 text-xs text-ink-muted">
                    ...
                  </div>
                )}
                <p className="text-xs text-ink-secondary">ATS baseline</p>
              </div>
              <div className="flex flex-col items-center gap-1.5">
                {typeof health.career_health_score === "number" ? (
                  <ScoreRing value={health.career_health_score} size={64} strokeWidth={6} />
                ) : (
                  <div className="flex size-16 items-center justify-center rounded-full bg-surface-2 text-xs text-ink-muted">
                    ...
                  </div>
                )}
                <p className="text-xs text-ink-secondary">Career health</p>
              </div>
            </div>
            {health.ats_baseline_missing_skills && health.ats_baseline_missing_skills.length > 0 && (
              <div>
                <div className="flex items-center justify-between gap-2">
                  <p className="text-xs font-medium text-ink-secondary">Missing skills for your target roles</p>
                  <FixSkillGapsDialog resumeVersionId={health.resume_version_id} label="Fix missing skills" />
                </div>
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {health.ats_baseline_missing_skills.slice(0, 8).map((skill) => (
                    <Badge key={skill} tone="warning">
                      {skill}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
