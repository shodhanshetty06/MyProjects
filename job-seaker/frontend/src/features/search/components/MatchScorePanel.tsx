import { useQuery } from "@tanstack/react-query";

import { Dialog } from "@/components/ui/Dialog";
import { SkeletonText } from "@/components/ui/Skeleton";
import { getWhyScore } from "@/features/search/api";

interface MatchScorePanelProps {
  jobId: string;
  jobTitle: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function Section({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-wide text-ink-muted">{label}</p>
      <p className="mt-0.5 text-sm text-ink">{value}</p>
    </div>
  );
}

/** "Why this score?" — a `Dialog` triggered from `JobCard`'s clickable `ScoreRing`, lazy-fetching `/search/jobs/{id}/why-score`. */
export function MatchScorePanel({ jobId, jobTitle, open, onOpenChange }: MatchScorePanelProps) {
  const whyScoreQuery = useQuery({
    queryKey: ["why-score", jobId],
    queryFn: () => getWhyScore(jobId),
    enabled: open,
  });

  const data = whyScoreQuery.data;

  return (
    <Dialog open={open} onOpenChange={onOpenChange} title="Why this score?" description={jobTitle} className="max-w-lg">
      {whyScoreQuery.isLoading ? (
        <SkeletonText lines={6} />
      ) : whyScoreQuery.isError || !data ? (
        <p className="text-sm text-ink-secondary">Couldn't load the match breakdown. Try again in a moment.</p>
      ) : (
        <div className="max-h-[60vh] space-y-4 overflow-y-auto pr-1">
          {data.narrative && (
            <p className="rounded-lg bg-surface-secondary px-3 py-2 text-sm italic text-ink-secondary">
              {data.narrative}
            </p>
          )}

          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-ink-muted">Skill match</p>
            <div className="mt-1 flex flex-wrap gap-1.5">
              {data.skill_checklist.length === 0 ? (
                <span className="text-xs text-ink-muted">No required skills listed</span>
              ) : (
                data.skill_checklist.map(({ skill, matched }) => (
                  <span
                    key={skill}
                    className={
                      matched
                        ? "rounded-full bg-success-soft px-2 py-0.5 text-xs font-medium text-success"
                        : "rounded-full bg-danger-soft px-2 py-0.5 text-xs font-medium text-danger"
                    }
                  >
                    {skill} {matched ? "✔" : "✖"}
                  </span>
                ))
              )}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <Section label="Experience match" value={data.experience_match} />
            <Section label="Education match" value={data.education_match} />
            <Section label="Location match" value={data.location_match} />
            <Section label="Salary match" value={data.salary_match} />
          </div>

          {data.resume_suggestions.length > 0 && (
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-ink-muted">Resume suggestions</p>
              <ul className="mt-1 list-inside list-disc space-y-1 text-sm text-ink-secondary">
                {data.resume_suggestions.map((suggestion, i) => (
                  <li key={i}>{suggestion}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </Dialog>
  );
}
