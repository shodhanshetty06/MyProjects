import { Lightbulb } from "lucide-react";
import { useMemo } from "react";

import { useHomeSummary } from "@/features/search/hooks";
import type { JobSearchHomeSummary } from "@/features/search/types";

/**
 * Insight strings derived from `GET /search/home-summary`, which computes
 * them over the user's full computed match set (bounded by
 * `MATCHING_CANDIDATE_LIMIT`, same as every other home-summary stat) —
 * not just the feed page currently scrolled into view. This replaced a
 * prior client-side-only version that recomputed the same shape of stats
 * from just the loaded `JobFeedItem[]` page, which drifted depending on
 * scroll position and active tab.
 */
function computeInsights(summary: JobSearchHomeSummary): string[] {
  const insights: string[] = [];

  if (summary.strong_match_count > 0) {
    insights.push(
      `${summary.strong_match_count} of your matched jobs are a strong fit (80%+)${
        summary.average_match_pct != null ? `, averaging ${Math.round(summary.average_match_pct)}% overall` : ""
      }.`,
    );
  } else if (summary.average_match_pct != null) {
    insights.push(`Your matched jobs average a ${Math.round(summary.average_match_pct)}% match — try broadening your preferences for more options.`);
  }

  if (summary.top_missing_skill && summary.top_missing_skill_count > 1) {
    insights.push(
      `"${summary.top_missing_skill}" is the most frequently missing skill across ${summary.top_missing_skill_count} of your matched jobs.`,
    );
  }

  if (summary.easy_apply_count > 0) {
    insights.push(`${summary.easy_apply_count} job${summary.easy_apply_count === 1 ? "" : "s"} in your matches look${summary.easy_apply_count === 1 ? "s" : ""} easy to apply to.`);
  }

  if (summary.applications_needing_follow_up_count > 0) {
    insights.push(
      `${summary.applications_needing_follow_up_count} application${summary.applications_needing_follow_up_count === 1 ? "" : "s"} could use a follow-up.`,
    );
  }

  return insights.slice(0, 3);
}

/** A thin strip of quick, backend-derived insight callouts above the job grid. */
export function AIInsightsStrip() {
  const summaryQuery = useHomeSummary();
  const insights = useMemo(() => (summaryQuery.data ? computeInsights(summaryQuery.data) : []), [summaryQuery.data]);

  if (insights.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-2">
      {insights.map((text, i) => (
        <div
          key={i}
          className="flex items-center gap-1.5 rounded-full border border-brand-border bg-brand-soft px-3 py-1.5 text-xs font-medium text-brand"
        >
          <Lightbulb className="size-3.5 shrink-0" aria-hidden />
          {text}
        </div>
      ))}
    </div>
  );
}
