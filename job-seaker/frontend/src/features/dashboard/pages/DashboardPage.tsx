import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Award, Briefcase, FileText, HeartPulse } from "lucide-react";
import { useState } from "react";

import { PageHeader } from "@/components/layout/PageHeader";
import { ResizableSplit } from "@/components/layout/ResizableSplit";
import { listATSResumes } from "@/features/ats-resume-generator";
import { computeJobReadiness } from "@/features/ai-career-coach";
import { useAuth } from "@/features/auth";
import { PinnedItemsPanel } from "@/features/dashboard/components/PinnedItemsPanel";
import { QuickActions } from "@/features/dashboard/components/QuickActions";
import { RecentActivityList } from "@/features/dashboard/components/RecentActivityList";
import { ScoreTile } from "@/features/dashboard/components/ScoreTile";
import { SuggestionsFeed } from "@/features/dashboard/components/SuggestionsFeed";
import { WeeklyGoalCard } from "@/features/dashboard/components/WeeklyGoalCard";
import { dismissRecommendation, listRecommendations } from "@/features/matching";
import type { Recommendation, RecommendationType } from "@/features/matching/types";
import { listResumes } from "@/features/resume-parser";
import { getTrackingDashboard } from "@/features/tracking";

export function DashboardPage() {
  const { user } = useAuth();
  const userId = user!.id;
  const queryClient = useQueryClient();
  const [suggestionType, setSuggestionType] = useState<RecommendationType>("job");

  const atsResumesQuery = useQuery({
    queryKey: ["ats-resumes", userId, "dashboard"],
    queryFn: () => listATSResumes(userId, 1, 5),
  });

  const resumesQuery = useQuery({
    queryKey: ["resumes", userId, "dashboard"],
    queryFn: () => listResumes(userId, 1, 10),
  });

  const trackingDashboardQuery = useQuery({
    queryKey: ["tracking-dashboard", userId],
    queryFn: () => getTrackingDashboard(userId),
    retry: false,
  });

  const recommendationsQuery = useQuery({
    queryKey: ["recommendations", userId, suggestionType],
    queryFn: () => listRecommendations(userId, suggestionType),
  });

  const parsedResumeVersionId = resumesQuery.data?.data.find(
    (resume) => resume.current_version?.status === "parsed",
  )?.current_version?.id;

  const jobReadinessQuery = useQuery({
    queryKey: ["job-readiness", userId, parsedResumeVersionId],
    queryFn: () => computeJobReadiness(userId, parsedResumeVersionId as string, null),
    enabled: false,
    staleTime: 30 * 60 * 1000,
  });

  const currentAtsVersion = atsResumesQuery.data?.data.find((r) => r.current_version)?.current_version;
  const resumeScore = currentAtsVersion?.ats_score ?? null;

  const statistics = trackingDashboardQuery.data?.statistics;
  const applicationHealth =
    statistics && statistics.total_applications > 0
      ? Math.min(100, Math.round(((statistics.interview_rate + statistics.offer_rate) / 2) * 100))
      : null;

  const interviewReadiness = jobReadinessQuery.data?.overall_score ?? null;

  const healthInputs = [resumeScore, applicationHealth, interviewReadiness].filter(
    (v): v is number => typeof v === "number",
  );
  const careerHealth = healthInputs.length > 0 ? Math.round(healthInputs.reduce((a, b) => a + b, 0) / healthInputs.length) : null;

  async function handleDismiss(id: string) {
    await dismissRecommendation(userId, id);
    queryClient.setQueryData<Recommendation[]>(["recommendations", userId, suggestionType], (prev) =>
      prev?.filter((rec) => rec.id !== id),
    );
  }

  return (
    <div>
      <PageHeader
        eyebrow="Workspace"
        title="Welcome back"
        description="Here's where your search stands today."
      />

      <div className="mb-6">
        <QuickActions />
      </div>

      <div className="mb-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <ScoreTile
          icon={FileText}
          label="Resume score"
          hint="ATS optimization score"
          isLoading={atsResumesQuery.isLoading}
          value={resumeScore}
          emptyLabel="Generate an ATS resume to see this"
        />
        <ScoreTile
          icon={Briefcase}
          label="Application health"
          hint="Interview & offer rate blend"
          isLoading={trackingDashboardQuery.isLoading}
          value={applicationHealth}
          emptyLabel="Track applications to see this"
        />
        <ScoreTile
          icon={Award}
          label="Interview readiness"
          hint="AI job-readiness score"
          value={interviewReadiness}
          isLoading={jobReadinessQuery.isFetching}
          onCompute={parsedResumeVersionId ? () => jobReadinessQuery.refetch() : undefined}
          isComputing={jobReadinessQuery.isFetching}
          emptyLabel={parsedResumeVersionId ? undefined : "Upload a resume to see this"}
        />
        <ScoreTile
          icon={HeartPulse}
          label="Career health"
          hint="Composite of the scores above"
          value={careerHealth}
          emptyLabel="Fill in the scores above to see this"
        />
      </div>

      <div className="mb-6">
        <WeeklyGoalCard
          applicationsThisWeek={statistics?.applications_last_7_days}
          isLoading={trackingDashboardQuery.isLoading}
        />
      </div>

      <ResizableSplit
        persistKey="dashboard-main"
        leftDefaultSize={64}
        left={
          <div className="flex flex-col gap-4 pr-3">
            <SuggestionsFeed
              activeType={suggestionType}
              onTypeChange={setSuggestionType}
              recommendations={recommendationsQuery.data ?? []}
              isLoading={recommendationsQuery.isLoading}
              onDismiss={handleDismiss}
            />
            <RecentActivityList
              applications={trackingDashboardQuery.data?.recent_applications ?? []}
              isLoading={trackingDashboardQuery.isLoading}
            />
          </div>
        }
        right={
          <div className="pl-3">
            <PinnedItemsPanel />
          </div>
        }
      />
    </div>
  );
}
