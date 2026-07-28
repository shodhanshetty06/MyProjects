import { useMutation } from "@tanstack/react-query";
import { motion } from "framer-motion";
import {
  Banknote,
  Bookmark,
  BookmarkCheck,
  Building2,
  FileStack,
  MapPin,
  MessageCircleQuestion,
  Send,
  Sparkles,
  TrendingUp,
} from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";

import { fadeInUp } from "@/components/motion/variants";
import { Badge, type BadgeTone } from "@/components/ui/Badge";
import { ScoreRing } from "@/components/ui/ScoreRing";
import { useAuth } from "@/features/auth";
import { sendChatMessage } from "@/features/ai-career-coach/api";
import { AutoApplyDialog } from "@/features/auto-apply/components/AutoApplyDialog";
import { MatchScorePanel } from "@/features/search/components/MatchScorePanel";
import { useSaveJob, useUnsaveJob } from "@/features/search/hooks";
import type { ApplicationDifficulty, JobFeedItem, SearchResultItem } from "@/features/search/types";
import { createApplication } from "@/features/tracking/api";
import { cn } from "@/lib/cn";
import { formatEnumLabel, formatLocation, formatRelativeDate, formatSalaryRange } from "@/lib/format";
import { paths } from "@/routes/paths";
import { toast } from "@/stores/toastStore";

/**
 * `JobCard` renders either a full `JobFeedItem` (the normal case, from
 * `/search/feed`) or a plain `SearchResultItem` (e.g. `CompanyDetailPage`,
 * which sources jobs from `job_ingestion` and doesn't have match/AI-signal
 * data) — the extra fields are optional so both shapes satisfy this prop.
 */
type JobCardData = SearchResultItem &
  Partial<Pick<JobFeedItem, "description" | "source_url" | "match" | "ai_signals" | "is_saved" | "application_status">>;

interface JobCardProps {
  job: JobCardData;
  className?: string;
}

const DIFFICULTY_TONE: Record<ApplicationDifficulty, BadgeTone> = {
  easy: "success",
  moderate: "warning",
  hard: "danger",
};

function qualityScoreTone(score: number): BadgeTone {
  if (score >= 80) return "success";
  if (score >= 50) return "warning";
  return "danger";
}

export function JobCard({ job, className }: JobCardProps) {
  const { user } = useAuth();
  const [panelOpen, setPanelOpen] = useState(false);
  const salary = formatSalaryRange(job.salary_min, job.salary_max, job.salary_currency, job.salary_period);
  const score = job.match?.match_score ?? job.match_score ?? job.relevance_score ?? job.keyword_score;

  const saveMutation = useSaveJob();
  const unsaveMutation = useUnsaveJob();
  const isSaved = job.is_saved ?? false;
  const isSaving = saveMutation.isPending || unsaveMutation.isPending;

  const applyMutation = useMutation({
    mutationFn: () =>
      createApplication(user!.id, {
        job_id: job.id,
        company_name: job.company.name,
        job_title: job.title,
        job_url: job.source_url ?? "",
        location: formatLocation(job.location),
        source: "manual",
      }),
    onSuccess: (app) => toast.success("Application tracked", `${app.job_title} added to your tracker.`),
    onError: () => toast.danger("Couldn't track this application", "Try again in a moment."),
  });

  const askAiMutation = useMutation({
    mutationFn: () =>
      sendChatMessage(
        user!.id,
        `Tell me more about this job and whether I'm a good fit: "${job.title}" at ${job.company.name}. ${
          job.match?.missing_skills?.length ? `I'm missing these skills: ${job.match.missing_skills.join(", ")}.` : ""
        }`,
        null,
      ),
    onSuccess: () => toast.success("Sent to AI Coach", "Open the coach chat to see the response."),
    onError: () => toast.danger("Couldn't reach AI Coach", "Try again in a moment."),
  });

  function toggleSave(e: React.MouseEvent) {
    e.preventDefault();
    if (isSaving) return;
    if (isSaved) unsaveMutation.mutate(job.id);
    else saveMutation.mutate(job.id);
  }

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
        onClick={toggleSave}
        disabled={isSaving}
        className="absolute right-4 top-4 flex size-7 items-center justify-center rounded-[var(--radius-sm)] text-ink-muted opacity-0 transition-opacity hover:bg-surface-2 hover:text-ink group-hover:opacity-100 data-[saved=true]:opacity-100 disabled:cursor-wait"
        data-saved={isSaved}
        aria-label={isSaved ? "Unsave job" : "Save job"}
      >
        {isSaved ? <BookmarkCheck className="size-4 text-brand" aria-hidden /> : <Bookmark className="size-4" aria-hidden />}
      </button>

      <Link to={paths.jobDetail(job.id)} className="flex gap-3">
        <div className="flex size-11 shrink-0 items-center justify-center overflow-hidden rounded-[var(--radius-md)] bg-surface-2 text-ink-muted">
          {job.company.logo_url ? (
            <img
              src={job.company.logo_url}
              alt=""
              width={44}
              height={44}
              loading="lazy"
              decoding="async"
              className="size-full object-cover"
            />
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
            {typeof job.quality_score === "number" && (
              <Badge
                tone={qualityScoreTone(job.quality_score)}
                title={job.quality_reasons
                  .map((reason) => `${reason.positive ? "✓" : "✗"} ${reason.label}`)
                  .join("\n")}
              >
                Quality {job.quality_score}/100
              </Badge>
            )}
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

        <div className="flex shrink-0 flex-col items-end justify-between gap-1">
          {typeof score === "number" && score > 0 && (
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                setPanelOpen(true);
              }}
              aria-label="Why this score?"
            >
              <ScoreRing value={Math.round(score)} size={40} strokeWidth={4} />
            </button>
          )}
          <span className="mt-auto text-[11px] text-ink-muted">{formatRelativeDate(job.posted_at)}</span>
        </div>
      </Link>

      {(job.match || job.ai_signals) && (
        <div className="mt-2.5 space-y-2 border-t border-border pt-2.5">
          {job.match?.strengths?.[0] && (
            <p className="flex items-start gap-1.5 text-[11px] leading-snug text-brand">
              <Sparkles className="mt-0.5 size-3 shrink-0" aria-hidden />
              {job.match.strengths[0]}
            </p>
          )}

          {job.match && job.match.missing_skills.length > 0 && (
            <div className="flex flex-wrap items-center gap-1">
              <span className="text-[10px] font-medium uppercase tracking-wide text-ink-muted">Missing:</span>
              {job.match.missing_skills.slice(0, 4).map((skill) => (
                <Badge key={skill} tone="danger" className="text-[10px]">
                  {skill}
                </Badge>
              ))}
            </div>
          )}

          {job.ai_signals && (
            <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
              <Badge tone={DIFFICULTY_TONE[job.ai_signals.application_difficulty]} className="text-[10px]">
                {formatEnumLabel(job.ai_signals.application_difficulty)} to apply
              </Badge>
              {job.ai_signals.expected_ats_score != null && (
                <Badge tone="info" className="text-[10px]">
                  Expected ATS {Math.round(job.ai_signals.expected_ats_score)}
                </Badge>
              )}
              <span className="flex items-center gap-1 text-ink-muted">
                <TrendingUp className="size-3" aria-hidden />
                {Math.round(job.ai_signals.hiring_probability)}% hiring probability
              </span>
            </div>
          )}
        </div>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-1.5 border-t border-border pt-2.5">
        <Link
          to={`${paths.atsGenerator}?target_job_id=${encodeURIComponent(job.id)}&job_title=${encodeURIComponent(job.title)}&company_name=${encodeURIComponent(job.company.name)}`}
          className="flex items-center gap-1 rounded-[var(--radius-sm)] px-2 py-1 text-[11px] font-medium text-ink-secondary transition-colors hover:bg-surface-2 hover:text-ink"
        >
          <FileStack className="size-3" aria-hidden />
          Generate ATS Resume
        </Link>
        <button
          type="button"
          onClick={(e) => {
            e.preventDefault();
            applyMutation.mutate();
          }}
          disabled={applyMutation.isPending}
          className="flex items-center gap-1 rounded-[var(--radius-sm)] px-2 py-1 text-[11px] font-medium text-ink-secondary transition-colors hover:bg-surface-2 hover:text-ink disabled:opacity-50"
        >
          <Send className="size-3" aria-hidden />
          Apply
        </button>
        <AutoApplyDialog
          jobId={job.id}
          jobTitle={job.title}
          companyName={job.company.name}
          jobDescription={job.description ?? ""}
          size="sm"
          className="h-auto gap-1 rounded-[var(--radius-sm)] px-2 py-1 text-[11px] shadow-none"
        />
        <button
          type="button"
          onClick={(e) => {
            e.preventDefault();
            askAiMutation.mutate();
          }}
          disabled={askAiMutation.isPending}
          className="flex items-center gap-1 rounded-[var(--radius-sm)] px-2 py-1 text-[11px] font-medium text-ink-secondary transition-colors hover:bg-surface-2 hover:text-ink disabled:opacity-50"
        >
          <MessageCircleQuestion className="size-3" aria-hidden />
          Ask AI
        </button>
        <Link
          to={paths.jobDetail(job.id)}
          className="ml-auto rounded-[var(--radius-sm)] px-2 py-1 text-[11px] font-medium text-brand transition-colors hover:bg-brand-soft"
        >
          View Details
        </Link>
      </div>

      <MatchScorePanel jobId={job.id} jobTitle={job.title} open={panelOpen} onOpenChange={setPanelOpen} />
    </motion.div>
  );
}
