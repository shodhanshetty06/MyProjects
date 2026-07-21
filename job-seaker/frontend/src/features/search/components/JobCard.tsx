import { motion } from "framer-motion";
import { Banknote, Building2, MapPin, Pin, PinOff, Sparkles } from "lucide-react";
import { Link } from "react-router-dom";

import { fadeInUp } from "@/components/motion/variants";
import { Badge } from "@/components/ui/Badge";
import { ScoreRing } from "@/components/ui/ScoreRing";
import type { SearchResultItem } from "@/features/search/types";
import { cn } from "@/lib/cn";
import { formatEnumLabel, formatLocation, formatRelativeDate, formatSalaryRange } from "@/lib/format";
import { paths } from "@/routes/paths";
import { usePinsStore } from "@/stores/pinsStore";

interface JobCardProps {
  job: SearchResultItem;
  className?: string;
}

export function JobCard({ job, className }: JobCardProps) {
  const isPinned = usePinsStore((state) => state.isPinned("job", job.id));
  const togglePin = usePinsStore((state) => state.togglePin);
  const salary = formatSalaryRange(job.salary_min, job.salary_max, job.salary_currency, job.salary_period);
  const score = job.match_score ?? job.relevance_score ?? job.keyword_score;

  return (
    <motion.div
      variants={fadeInUp}
      className={cn(
        "group relative rounded-[var(--radius-lg)] border border-border bg-surface p-4 transition-all hover:-translate-y-0.5 hover:border-brand-border hover:shadow-sm",
        className,
      )}
    >
      <button
        type="button"
        onClick={(e) => {
          e.preventDefault();
          togglePin({ id: job.id, type: "job", label: job.title, meta: job.company.name });
        }}
        className="absolute right-4 top-4 flex size-7 items-center justify-center rounded-[var(--radius-sm)] text-ink-muted opacity-0 transition-opacity hover:bg-surface-2 hover:text-ink group-hover:opacity-100 data-[pinned=true]:opacity-100"
        data-pinned={isPinned}
        aria-label={isPinned ? "Unpin job" : "Pin job"}
      >
        {isPinned ? <PinOff className="size-4 text-brand" aria-hidden /> : <Pin className="size-4" aria-hidden />}
      </button>

      <Link to={paths.jobDetail(job.id)} className="flex gap-3">
        <div className="flex size-11 shrink-0 items-center justify-center overflow-hidden rounded-[var(--radius-md)] bg-surface-2 text-ink-muted">
          {job.company.logo_url ? (
            <img src={job.company.logo_url} alt="" className="size-full object-cover" />
          ) : (
            <Building2 className="size-5" aria-hidden />
          )}
        </div>

        <div className="min-w-0 flex-1 pr-8">
          <p className="truncate text-sm font-semibold text-ink">{job.title}</p>
          <p className="truncate text-xs text-ink-secondary">{job.company.name}</p>

          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-muted">
            <span className="flex items-center gap-1">
              <MapPin className="size-3" aria-hidden />
              {formatLocation(job.location)}
            </span>
            <Badge tone="neutral">{formatEnumLabel(job.work_mode)}</Badge>
            <Badge tone="neutral">{formatEnumLabel(job.employment_type)}</Badge>
            {job.experience_level && <Badge tone="neutral">{formatEnumLabel(job.experience_level)}</Badge>}
          </div>

          {salary && (
            <p className="mt-2 flex items-center gap-1 text-xs font-medium text-ink">
              <Banknote className="size-3 text-success" aria-hidden />
              {salary}
            </p>
          )}

          {job.skills.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {job.skills.slice(0, 5).map((skill) => (
                <Badge key={skill.id} tone="neutral" className="text-[10px]">
                  {skill.name}
                </Badge>
              ))}
              {job.skills.length > 5 && (
                <Badge tone="neutral" className="text-[10px]">
                  +{job.skills.length - 5}
                </Badge>
              )}
            </div>
          )}
        </div>

        <div className="flex shrink-0 flex-col items-end justify-between">
          {typeof score === "number" && score > 0 && (
            <ScoreRing value={Math.round(score)} size={40} strokeWidth={4} />
          )}
          <span className="mt-auto text-[11px] text-ink-muted">{formatRelativeDate(job.posted_at)}</span>
        </div>
      </Link>

      {typeof job.match_score === "number" && (
        <div className="mt-2 flex items-center gap-1 text-[11px] font-medium text-brand">
          <Sparkles className="size-3" aria-hidden />
          AI matched
        </div>
      )}
    </motion.div>
  );
}
