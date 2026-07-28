import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Navigate } from "react-router-dom";

import { PageHeader } from "@/components/layout/PageHeader";
import { ResizableSplit } from "@/components/layout/ResizableSplit";
import { ErrorState } from "@/components/ui/ErrorState";
import { RecentActivityList } from "@/features/dashboard/components/RecentActivityList";
import { useAuth } from "@/features/auth";
import { dismissRecommendation } from "@/features/matching";
import type { Recommendation } from "@/features/matching/types";
import { getWorkspace } from "@/features/workspace/api";
import { AICoachPanel } from "@/features/workspace/components/AICoachPanel";
import { NextBestActionsList } from "@/features/workspace/components/NextBestActionsList";
import { NotificationsPanel } from "@/features/workspace/components/NotificationsPanel";
import { PipelineBanner } from "@/features/workspace/components/PipelineBanner";
import { RecommendedJobsPanel } from "@/features/workspace/components/RecommendedJobsPanel";
import { ResumeHealthPanel } from "@/features/workspace/components/ResumeHealthPanel";
import { ResumeVersionsPanel } from "@/features/workspace/components/ResumeVersionsPanel";
import { useWorkspaceSocket } from "@/features/workspace/useWorkspaceSocket";
import { paths } from "@/routes/paths";

const WORKSPACE_QUERY_KEY = ["workspace"];

const EMPTY_RESUME_HEALTH = {
  resume_id: null,
  resume_version_id: null,
  resume_score: null,
  resume_score_breakdown: null,
  ats_baseline_score: null,
  ats_baseline_missing_skills: null,
  health_computed_at: null,
  career_health_score: null,
} as const;

/**
 * The single integrated AI Workspace — replaces manually visiting Resume
 * Analysis / ATS Analysis / Manual Recommendations as separate pages.
 * Everything here comes from one `GET /workspace` aggregate and refreshes
 * itself automatically: `useWorkspaceSocket` invalidates the query on every
 * pipeline/match/recommendation/application event, so uploading a new
 * resume or a background job match updates this page with no manual
 * refresh button anywhere.
 */
export function WorkspacePage() {
  const { user } = useAuth();
  const userId = user!.id;
  const queryClient = useQueryClient();

  const workspaceQuery = useQuery({
    queryKey: WORKSPACE_QUERY_KEY,
    queryFn: getWorkspace,
  });

  useWorkspaceSocket(userId, () => {
    void queryClient.invalidateQueries({ queryKey: WORKSPACE_QUERY_KEY });
  });

  if (!workspaceQuery.isLoading && workspaceQuery.data && !workspaceQuery.data.onboarding_completed) {
    return <Navigate to={paths.onboarding} replace />;
  }

  if (workspaceQuery.isError) {
    return <ErrorState description="Couldn't load your workspace. Try refreshing the page." />;
  }

  const data = workspaceQuery.data;
  const isLoading = workspaceQuery.isLoading;
  const recommendedJobs = data?.recommended_jobs ?? [];

  async function handleDismiss(id: string) {
    await dismissRecommendation(userId, id);
    queryClient.setQueryData(WORKSPACE_QUERY_KEY, (prev: typeof data) =>
      prev
        ? {
            ...prev,
            recommended_jobs: prev.recommended_jobs.filter((rec: Recommendation) => rec.id !== id),
          }
        : prev,
    );
  }

  return (
    <div>
      <PageHeader
        eyebrow="Workspace"
        title="Your AI workspace"
        description="Everything about your search in one place — it updates itself as things finish processing."
      />

      {data?.pipeline_status && data.pipeline_status.status !== "completed" && (
        <PipelineBanner status={data.pipeline_status} />
      )}

      <div className="mb-6">
        <ResumeHealthPanel health={data?.resume_health ?? EMPTY_RESUME_HEALTH} isLoading={isLoading} />
      </div>

      <ResizableSplit
        persistKey="workspace-main"
        leftDefaultSize={64}
        left={
          <div className="flex flex-col gap-4 pr-3">
            <RecommendedJobsPanel
              recommendations={recommendedJobs}
              isLoading={isLoading}
              onDismiss={handleDismiss}
            />
            <RecentActivityList
              applications={data?.applications.recent_applications ?? []}
              isLoading={isLoading}
            />
            <ResumeVersionsPanel
              resumeId={data?.resume_health.resume_id ?? null}
              versions={data?.resume_versions ?? []}
              isLoading={isLoading}
            />
          </div>
        }
        right={
          <div className="flex flex-col gap-4 pl-3">
            <NextBestActionsList
              actions={data?.next_best_actions ?? []}
              resumeHealth={data?.resume_health}
              isLoading={isLoading}
            />
            <AICoachPanel insights={data?.coach_highlights ?? []} isLoading={isLoading} />
            <NotificationsPanel
              notifications={data?.notifications ?? []}
              unreadCount={data?.unread_notification_count ?? 0}
              isLoading={isLoading}
            />
          </div>
        }
      />
    </div>
  );
}
