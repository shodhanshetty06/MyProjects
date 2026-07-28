import { motion } from "framer-motion";
import { ArrowRight, Building2, Sparkles, TrendingUp } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";

import { fadeInUp } from "@/components/motion/variants";
import { Skeleton } from "@/components/ui/Skeleton";
import { useHomeSummary } from "@/features/search/hooks";
import { paths } from "@/routes/paths";

interface StatProps {
  label: string;
  value: ReactNode;
}

function Stat({ label, value }: StatProps) {
  return (
    <div className="min-w-0">
      <p className="text-lg font-semibold text-ink">{value}</p>
      <p className="truncate text-xs text-ink-muted">{label}</p>
    </div>
  );
}

/** Greeting + at-a-glance counts + the platform's top next-best-action, sourced from `GET /search/home-summary`. */
export function TopSummaryBar() {
  const summaryQuery = useHomeSummary();

  if (summaryQuery.isLoading) {
    return (
      <div className="rounded-[var(--radius-lg)] border border-border bg-surface p-5">
        <Skeleton className="h-5 w-64" />
        <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-10 w-full" />
          ))}
        </div>
      </div>
    );
  }

  if (summaryQuery.isError || !summaryQuery.data) {
    return null;
  }

  const summary = summaryQuery.data;
  const action = summary.recommended_action;

  return (
    <motion.div
      variants={fadeInUp}
      initial="hidden"
      animate="visible"
      className="rounded-[var(--radius-lg)] border border-border bg-surface p-5"
    >
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="flex items-center gap-1.5 text-base font-semibold text-ink">
            <Sparkles className="size-4 text-brand" aria-hidden />
            {summary.greeting}
          </h2>
          <p className="mt-0.5 text-sm text-ink-secondary">
            {summary.jobs_matching_profile_count.toLocaleString()} jobs match your profile right now.
          </p>
        </div>

        {action && (
          <Link
            to={action.cta_path}
            className="group flex shrink-0 items-center gap-2 rounded-[var(--radius-md)] border border-brand-border bg-brand-soft px-3.5 py-2 text-sm font-medium text-brand transition-colors hover:bg-brand-soft/70"
          >
            {action.title}
            <ArrowRight className="size-3.5 transition-transform group-hover:translate-x-0.5" aria-hidden />
          </Link>
        )}
      </div>

      <div className="mt-5 grid grid-cols-2 gap-4 border-t border-border pt-4 sm:grid-cols-4 lg:grid-cols-5">
        <Stat label="New jobs since yesterday" value={summary.new_jobs_since_yesterday.toLocaleString()} />
        <Stat label="Average match" value={`${Math.round(summary.average_match_pct)}%`} />
        <Stat label="Highest match" value={`${Math.round(summary.highest_match_pct)}%`} />
        <Stat
          label="Companies hiring now"
          value={
            <span className="inline-flex items-center gap-1">
              <Building2 className="size-3.5 text-ink-muted" aria-hidden />
              {summary.companies_hiring_now.toLocaleString()}
            </span>
          }
        />
        {summary.applications_needing_follow_up_count > 0 && (
          <Link to={paths.tracker} className="min-w-0 transition-opacity hover:opacity-80">
            <Stat
              label="Applications needing follow-up"
              value={<span className="text-warning">{summary.applications_needing_follow_up_count.toLocaleString()}</span>}
            />
          </Link>
        )}
      </div>

      {action && (
        <p className="mt-3 flex items-center gap-1.5 text-xs text-ink-muted">
          <TrendingUp className="size-3.5" aria-hidden />
          {action.description}
        </p>
      )}
    </motion.div>
  );
}
