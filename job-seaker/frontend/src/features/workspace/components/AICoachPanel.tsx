import { motion } from "framer-motion";
import { Bot } from "lucide-react";
import { Link } from "react-router-dom";

import { staggerChildren } from "@/components/motion/variants";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { SkeletonText } from "@/components/ui/Skeleton";
import type { CareerInsightSummary, CareerInsightType } from "@/features/ai-career-coach/types";
import { formatRelativeDate } from "@/lib/format";
import { paths } from "@/routes/paths";

const INSIGHT_LABELS: Record<CareerInsightType, string> = {
  resume_review: "Resume review",
  career_suggestions: "Career suggestions",
  skill_gap_analysis: "Skill gap analysis",
  learning_recommendations: "Learning recommendations",
  salary_prediction: "Salary prediction",
  career_roadmap: "Career roadmap",
  interview_preparation: "Interview preparation",
  job_readiness_score: "Job readiness score",
  market_trend_analysis: "Market trend analysis",
  country_demand: "Country demand",
};

interface AICoachPanelProps {
  insights: CareerInsightSummary[];
  isLoading: boolean;
}

export function AICoachPanel({ insights, isLoading }: AICoachPanelProps) {
  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-3 space-y-0">
        <CardTitle className="flex items-center gap-1.5">
          <Bot className="size-4 text-brand" aria-hidden />
          AI Career Coach
        </CardTitle>
        <Link to={paths.coach} className="text-xs font-medium text-brand hover:underline">
          Open coach
        </Link>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <SkeletonText lines={2} />
        ) : insights.length === 0 ? (
          <EmptyState
            icon={Bot}
            title="No insights yet"
            description="Ask your AI Career Coach for a resume review, skill gap analysis, or salary estimate."
            action={
              <Link to={paths.coach} className="text-sm font-medium text-brand hover:underline">
                Talk to your coach
              </Link>
            }
          />
        ) : (
          <motion.ul variants={staggerChildren(0.05)} initial="hidden" animate="visible" className="space-y-1">
            {insights.map((insight) => (
              <motion.li key={insight.id} variants={{ hidden: { opacity: 0 }, visible: { opacity: 1 } }}>
                <Link
                  to={paths.coachInsights}
                  className="flex items-center justify-between gap-3 rounded-[var(--radius-md)] px-2 py-2 transition-colors hover:bg-surface-2"
                >
                  <p className="truncate text-sm font-medium text-ink">
                    {INSIGHT_LABELS[insight.insight_type] ?? insight.insight_type}
                  </p>
                  <span className="shrink-0 text-xs text-ink-muted">
                    {formatRelativeDate(insight.created_at)}
                  </span>
                </Link>
              </motion.li>
            ))}
          </motion.ul>
        )}
      </CardContent>
    </Card>
  );
}
