import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { History } from "lucide-react";
import { useState } from "react";

import { PageHeader } from "@/components/layout/PageHeader";
import { staggerChildren } from "@/components/motion/variants";
import { Badge } from "@/components/ui/Badge";
import { Card, CardContent } from "@/components/ui/Card";
import { Dialog } from "@/components/ui/Dialog";
import { EmptyState } from "@/components/ui/EmptyState";
import { Pagination } from "@/components/ui/Pagination";
import { SkeletonText } from "@/components/ui/Skeleton";
import { useAuth } from "@/features/auth";
import { getInsight, listInsights } from "@/features/ai-career-coach/api";
import {
  CareerSuggestionsView,
  CountryDemandView,
  InterviewPrepView,
  JobReadinessView,
  LearningRecommendationsView,
  MarketTrendsView,
  ResumeReviewView,
  RoadmapView,
  SalaryPredictionView,
  SkillGapView,
} from "@/features/ai-career-coach/components/ToolResults";
import type { CareerInsightType } from "@/features/ai-career-coach/types";
import { formatEnumLabel, formatRelativeDate } from "@/lib/format";

export function CoachInsightsPage() {
  const { user } = useAuth();
  const userId = user!.id;
  const [page, setPage] = useState(1);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const insightsQuery = useQuery({
    queryKey: ["coach-insights", userId, page],
    queryFn: () => listInsights(userId, undefined, page, 20),
  });

  const insightDetailQuery = useQuery({
    queryKey: ["coach-insight", userId, selectedId],
    queryFn: () => getInsight(userId, selectedId as string),
    enabled: !!selectedId,
  });

  const insights = insightsQuery.data?.data ?? [];

  return (
    <div className="mx-auto max-w-2xl">
      <PageHeader eyebrow="AI Career Coach" title="Insight history" description="Every tool result you've generated, saved for later." />

      {insightsQuery.isLoading ? (
        <SkeletonText lines={5} />
      ) : insights.length === 0 ? (
        <EmptyState icon={History} title="No insights yet" description="Run a coach tool to see results here." />
      ) : (
        <>
          <motion.div variants={staggerChildren(0.04)} initial="hidden" animate="visible" className="space-y-2">
            {insights.map((insight) => (
              <motion.div key={insight.id} variants={{ hidden: { opacity: 0, y: 6 }, visible: { opacity: 1, y: 0 } }}>
                <Card
                  className="cursor-pointer p-4 transition-colors hover:bg-surface-2"
                  onClick={() => setSelectedId(insight.id)}
                >
                  <CardContent className="flex items-center justify-between p-0">
                    <Badge tone="brand">{formatEnumLabel(insight.insight_type)}</Badge>
                    <span className="text-xs text-ink-muted">{formatRelativeDate(insight.created_at)}</span>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </motion.div>
          {insightsQuery.data && (
            <Pagination
              page={insightsQuery.data.pagination.page}
              totalPages={insightsQuery.data.pagination.total_pages}
              totalItems={insightsQuery.data.pagination.total_items}
              onPageChange={setPage}
            />
          )}
        </>
      )}

      <Dialog
        open={!!selectedId}
        onOpenChange={(open) => !open && setSelectedId(null)}
        title={insightDetailQuery.data ? formatEnumLabel(insightDetailQuery.data.insight_type) : "Insight"}
        className="max-w-xl"
      >
        {insightDetailQuery.isLoading ? <SkeletonText lines={5} /> : insightDetailQuery.data && <InsightBody insight={insightDetailQuery.data} />}
      </Dialog>
    </div>
  );
}

function InsightBody({ insight }: { insight: { insight_type: CareerInsightType; result: Record<string, unknown> } }) {
  const result = insight.result;
  switch (insight.insight_type) {
    case "resume_review":
      return <ResumeReviewView result={result as never} />;
    case "career_suggestions":
      return <CareerSuggestionsView result={result as never} />;
    case "skill_gap_analysis":
      return <SkillGapView result={result as never} />;
    case "learning_recommendations":
      return <LearningRecommendationsView result={result as never} />;
    case "salary_prediction":
      return <SalaryPredictionView result={result as never} />;
    case "career_roadmap":
      return <RoadmapView result={result as never} />;
    case "interview_preparation":
      return <InterviewPrepView result={result as never} />;
    case "job_readiness_score":
      return <JobReadinessView result={result as never} />;
    case "market_trend_analysis":
      return <MarketTrendsView result={result as never} />;
    case "country_demand":
      return <CountryDemandView result={result as never} />;
    default:
      return <p className="text-sm text-ink-muted">Unknown insight type.</p>;
  }
}
