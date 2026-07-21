import { useQuery } from "@tanstack/react-query";
import { Flame, TrendingUp } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { SkeletonText } from "@/components/ui/Skeleton";
import { getTrendingJobs, getTrendingSkills } from "@/features/search/api";
import { formatRelativeDate } from "@/lib/format";

interface TrendingPanelProps {
  onSkillClick: (skill: string) => void;
}

export function TrendingPanel({ onSkillClick }: TrendingPanelProps) {
  const skillsQuery = useQuery({ queryKey: ["trending-skills"], queryFn: () => getTrendingSkills(30, 12) });
  const jobsQuery = useQuery({ queryKey: ["trending-jobs"], queryFn: () => getTrendingJobs(30, 8) });

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-1.5">
            <Flame className="size-4 text-warning" aria-hidden />
            Trending skills
          </CardTitle>
        </CardHeader>
        <CardContent>
          {skillsQuery.isLoading ? (
            <SkeletonText lines={4} />
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {(skillsQuery.data ?? []).map((skill) => (
                <button
                  key={skill.skill_id}
                  type="button"
                  onClick={() => onSkillClick(skill.name)}
                  className="rounded-full border border-border bg-surface-2 px-2.5 py-1 text-xs font-medium text-ink-secondary transition-colors hover:border-brand-border hover:bg-brand-soft hover:text-brand"
                >
                  {skill.name} · {skill.job_count}
                </button>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-1.5">
            <TrendingUp className="size-4 text-brand" aria-hidden />
            Trending roles
          </CardTitle>
        </CardHeader>
        <CardContent>
          {jobsQuery.isLoading ? (
            <SkeletonText lines={4} />
          ) : (
            <ul className="space-y-2.5">
              {(jobsQuery.data ?? []).map((job) => (
                <li key={job.id} className="text-sm">
                  <p className="font-medium text-ink">{job.title}</p>
                  <p className="text-xs text-ink-muted">
                    {job.company_name} · {formatRelativeDate(job.posted_at)}
                  </p>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
