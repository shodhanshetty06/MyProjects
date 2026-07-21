import { motion } from "framer-motion";
import { Building2, Sparkles, X } from "lucide-react";

import { staggerChildren } from "@/components/motion/variants";
import { Badge } from "@/components/ui/Badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { SkeletonText } from "@/components/ui/Skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/Tabs";
import type { Recommendation, RecommendationType } from "@/features/matching/types";

interface SuggestionsFeedProps {
  activeType: RecommendationType;
  onTypeChange: (type: RecommendationType) => void;
  recommendations: Recommendation[];
  isLoading: boolean;
  onDismiss: (id: string) => void;
}

const TYPE_LABELS: Record<RecommendationType, string> = { job: "Jobs", company: "Companies", skill: "Skills" };

export function SuggestionsFeed({ activeType, onTypeChange, recommendations, isLoading, onDismiss }: SuggestionsFeedProps) {
  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-3 space-y-0">
        <CardTitle className="flex items-center gap-1.5">
          <Sparkles className="size-4 text-brand" aria-hidden />
          AI suggestions
        </CardTitle>
        <Tabs value={activeType} onValueChange={(v) => onTypeChange(v as RecommendationType)}>
          <TabsList>
            {(Object.keys(TYPE_LABELS) as RecommendationType[]).map((type) => (
              <TabsTrigger key={type} value={type} className="px-2.5 py-1 text-xs">
                {TYPE_LABELS[type]}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </CardHeader>
      <CardContent>
        <Tabs value={activeType}>
          <TabsContent value={activeType} forceMount>
            {isLoading ? (
              <SkeletonText lines={4} />
            ) : recommendations.length === 0 ? (
              <EmptyState
                icon={Sparkles}
                title="No suggestions yet"
                description="Upload a resume and set your preferences to get personalized AI recommendations."
              />
            ) : (
              <motion.ul variants={staggerChildren(0.05)} initial="hidden" animate="visible" className="space-y-2">
                {recommendations.map((rec) => (
                  <motion.li
                    key={rec.id}
                    variants={{ hidden: { opacity: 0, x: -8 }, visible: { opacity: 1, x: 0 } }}
                    className="flex items-start gap-3 rounded-[var(--radius-md)] border border-border p-3"
                  >
                    <div className="flex size-8 shrink-0 items-center justify-center rounded-[var(--radius-sm)] bg-surface-2 text-ink-muted">
                      <Building2 className="size-4" aria-hidden />
                    </div>
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
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
