import { motion } from "framer-motion";
import { Sparkles, X } from "lucide-react";

import { staggerChildren } from "@/components/motion/variants";
import { Badge } from "@/components/ui/Badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { SkeletonText } from "@/components/ui/Skeleton";
import type { Recommendation } from "@/features/matching/types";

interface RecommendedJobsPanelProps {
  recommendations: Recommendation[];
  isLoading: boolean;
  onDismiss: (id: string) => void;
}

/**
 * Job-only recommendations panel for the workspace — unlike the legacy
 * dashboard's tabbed `SuggestionsFeed` (jobs/companies/skills), the
 * workspace aggregate only carries job recommendations (see
 * `WorkspaceSummary.recommended_jobs`), so this renders a single flat list
 * rather than tabs that would always show empty company/skill panes.
 */
export function RecommendedJobsPanel({ recommendations, isLoading, onDismiss }: RecommendedJobsPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-1.5">
          <Sparkles className="size-4 text-brand" aria-hidden />
          Recommended jobs
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <SkeletonText lines={4} />
        ) : recommendations.length === 0 ? (
          <EmptyState
            icon={Sparkles}
            title="No recommendations yet"
            description="Once your resume finishes processing, matched jobs will show up here automatically."
          />
        ) : (
          <motion.ul variants={staggerChildren(0.05)} initial="hidden" animate="visible" className="space-y-2">
            {recommendations.map((rec) => (
              <motion.li
                key={rec.id}
                variants={{ hidden: { opacity: 0, x: -8 }, visible: { opacity: 1, x: 0 } }}
                className="flex items-start gap-3 rounded-[var(--radius-md)] border border-border p-3"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <p className="truncate text-sm font-medium text-ink">{rec.display_name}</p>
                    <Badge tone="brand">{Math.round(rec.score)}% match</Badge>
                  </div>
                  <p className="mt-0.5 line-clamp-2 text-xs text-ink-secondary">{rec.reason}</p>
                </div>
                <button
                  type="button"
                  onClick={() => onDismiss(rec.id)}
                  className="shrink-0 rounded-[var(--radius-sm)] p-1 text-ink-muted transition-colors hover:bg-surface-2 hover:text-ink"
                  aria-label={`Dismiss ${rec.display_name}`}
                >
                  <X className="size-3.5" aria-hidden />
                </button>
              </motion.li>
            ))}
          </motion.ul>
        )}
      </CardContent>
    </Card>
  );
}
