import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { RefreshCw, Sparkles, X } from "lucide-react";
import { Link } from "react-router-dom";

import { PageHeader } from "@/components/layout/PageHeader";
import { staggerChildren } from "@/components/motion/variants";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { SkeletonCard } from "@/components/ui/Skeleton";
import { useAuth } from "@/features/auth";
import { dismissRecommendation, listRecommendations, refreshRecommendations } from "@/features/matching/api";
import { paths } from "@/routes/paths";
import { toast } from "@/stores/toastStore";

export function RecommendedJobsPage() {
  const { user } = useAuth();
  const userId = user!.id;
  const queryClient = useQueryClient();

  const recommendationsQuery = useQuery({
    queryKey: ["recommendations", userId, "job", "full"],
    queryFn: () => listRecommendations(userId, "job"),
  });

  const refreshMutation = useMutation({
    mutationFn: () => refreshRecommendations(userId),
    onSuccess: () => {
      toast.success("Refreshing recommendations", "New matches will appear shortly.");
      void queryClient.invalidateQueries({ queryKey: ["recommendations", userId] });
    },
  });

  async function handleDismiss(id: string) {
    await dismissRecommendation(userId, id);
    void queryClient.invalidateQueries({ queryKey: ["recommendations", userId, "job", "full"] });
  }

  const recommendations = recommendationsQuery.data ?? [];

  return (
    <div>
      <PageHeader
        eyebrow="Job search"
        title="Recommended for you"
        description="AI-matched jobs based on your resume, preferences, and search activity."
        actions={
          <Button variant="secondary" onClick={() => refreshMutation.mutate()} isLoading={refreshMutation.isPending}>
            <RefreshCw className="size-4" aria-hidden />
            Refresh
          </Button>
        }
      />

      {recommendationsQuery.isLoading ? (
        <div className="grid gap-3 sm:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      ) : recommendations.length === 0 ? (
        <EmptyState
          icon={Sparkles}
          title="No recommendations yet"
          description="Upload a resume and set your match preferences to get personalized job recommendations."
        />
      ) : (
        <motion.div variants={staggerChildren(0.05)} initial="hidden" animate="visible" className="grid gap-3 sm:grid-cols-2">
          {recommendations.map((rec) => (
            <motion.div key={rec.id} variants={{ hidden: { opacity: 0, y: 8 }, visible: { opacity: 1, y: 0 } }}>
              <Card className="p-4">
                <div className="flex items-start justify-between gap-2">
                  {rec.job_id ? (
                    <Link to={paths.jobDetail(rec.job_id)} className="text-sm font-semibold text-ink hover:text-brand">
                      {rec.display_name}
                    </Link>
                  ) : (
                    <p className="text-sm font-semibold text-ink">{rec.display_name}</p>
                  )}
                  <Badge tone="brand">{Math.round(rec.score)}%</Badge>
                </div>
                <p className="mt-1.5 text-xs text-ink-secondary">{rec.reason}</p>
                <div className="mt-3 flex justify-end">
                  <button
                    type="button"
                    onClick={() => handleDismiss(rec.id)}
                    className="flex items-center gap-1 text-xs font-medium text-ink-muted hover:text-ink"
                  >
                    <X className="size-3" aria-hidden />
                    Dismiss
                  </button>
                </div>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      )}
    </div>
  );
}
