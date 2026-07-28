import { useMutation, useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  Banknote,
  Bookmark,
  BookmarkCheck,
  Briefcase,
  Building2,
  Calendar,
  CheckCircle2,
  Compass,
  DollarSign,
  ExternalLink,
  FileStack,
  ListChecks,
  MapPin,
  MessageCircleQuestion,
  Send,
  Sparkles,
  TrendingUp,
  XCircle,
} from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { fadeInUp } from "@/components/motion/variants";
import { Badge, type BadgeTone } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent } from "@/components/ui/Card";
import { Dialog } from "@/components/ui/Dialog";
import { ErrorState } from "@/components/ui/ErrorState";
import { ScoreRing } from "@/components/ui/ScoreRing";
import { SkeletonText } from "@/components/ui/Skeleton";
import { useAuth } from "@/features/auth";
import { sendChatMessage } from "@/features/ai-career-coach/api";
import { AutoApplyDialog } from "@/features/auto-apply/components/AutoApplyDialog";
import { MatchScorePanel } from "@/features/search/components/MatchScorePanel";
import { getJobDetail } from "@/features/search/api";
import { useSaveJob, useUnsaveJob } from "@/features/search/hooks";
import type { ApplicationDifficulty } from "@/features/search/types";
import { createApplication } from "@/features/tracking/api";
import { cn } from "@/lib/cn";
import { formatEnumLabel, formatLocation, formatRelativeDate, formatSalaryRange } from "@/lib/format";
import { paths } from "@/routes/paths";
import { toast } from "@/stores/toastStore";

const DIFFICULTY_TONE: Record<ApplicationDifficulty, BadgeTone> = {
  easy: "success",
  moderate: "warning",
  hard: "danger",
};

/**
 * Splits a plain-text job description into rough "Responsibilities" /
 * "Requirements" sections by looking for common section headers. This is a
 * light heuristic over existing `description` text, not a new AI call —
 * if no headers are found, the whole description is shown as one block.
 */
function splitDescription(description: string): { responsibilities: string | null; requirements: string | null; rest: string } {
  const reqMatch = description.match(/(requirements?|qualifications?)\s*[:\-]?\s*\n/i);
  const respMatch = description.match(/(responsibilities|what you.?ll do|role overview)\s*[:\-]?\s*\n/i);

  if (!reqMatch && !respMatch) {
    return { responsibilities: null, requirements: null, rest: description };
  }

  const respStart = respMatch ? respMatch.index! + respMatch[0].length : null;
  const reqStart = reqMatch ? reqMatch.index! + reqMatch[0].length : null;

  let responsibilities: string | null = null;
  let requirements: string | null = null;

  if (respStart !== null) {
    const end = reqMatch && reqMatch.index! > respStart ? reqMatch.index! : description.length;
    responsibilities = description.slice(respStart, end).trim();
  }
  if (reqStart !== null) {
    const end = respMatch && respMatch.index! > reqStart ? respMatch.index! : description.length;
    requirements = description.slice(reqStart, end).trim();
  }

  return { responsibilities, requirements, rest: description };
}

export function JobDetailPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const { user } = useAuth();
  const navigate = useNavigate();
  const [trackDialogOpen, setTrackDialogOpen] = useState(false);
  const [panelOpen, setPanelOpen] = useState(false);

  const jobQuery = useQuery({
    queryKey: ["job-detail", jobId],
    queryFn: () => getJobDetail(jobId as string),
    enabled: !!jobId,
  });

  const saveMutation = useSaveJob();
  const unsaveMutation = useUnsaveJob();

  const trackMutation = useMutation({
    mutationFn: () =>
      createApplication(user!.id, {
        job_id: jobQuery.data!.id,
        company_name: jobQuery.data!.company.name,
        job_title: jobQuery.data!.title,
        job_url: jobQuery.data!.source_url,
        location: formatLocation(jobQuery.data!.location),
        source: "manual",
      }),
    onSuccess: (app) => {
      toast.success("Application tracked", `${app.job_title} added to your tracker.`);
      setTrackDialogOpen(false);
      navigate(paths.trackerTimeline(app.id));
    },
    onError: () => toast.danger("Couldn't track this application", "Try again in a moment."),
  });

  const askAiMutation = useMutation({
    mutationFn: () => {
      const job = jobQuery.data!;
      const missing = job.match?.missing_skills?.length ? ` I'm missing these skills: ${job.match.missing_skills.join(", ")}.` : "";
      return sendChatMessage(
        user!.id,
        `Tell me more about this job and whether I'm a good fit: "${job.title}" at ${job.company.name}.${missing}`,
        null,
      );
    },
    onSuccess: () => toast.success("Sent to AI Coach", "Open the coach chat to see the response."),
    onError: () => toast.danger("Couldn't reach AI Coach", "Try again in a moment."),
  });

  const sections = useMemo(() => (jobQuery.data ? splitDescription(jobQuery.data.description) : null), [jobQuery.data]);

  // Salary insight computed client-side from this single job's own salary
  // range vs. its match score — a lightweight, documented substitute for a
  // real market-comparison feature (no new backend endpoint for this).
  const salaryInsight = useMemo(() => {
    const job = jobQuery.data;
    if (!job || (!job.salary_min && !job.salary_max)) return null;
    const score = job.match?.match_score;
    if (score == null) return "Salary listed by the employer.";
    if (score >= 80) return "Your profile is a strong match — this salary range is likely negotiable in your favor.";
    if (score >= 50) return "A moderate match — this salary is likely at or near the employer's stated range.";
    return "A lighter match — expect less room to negotiate above the stated range.";
  }, [jobQuery.data]);

  if (jobQuery.isLoading) {
    return (
      <div className="mx-auto max-w-3xl space-y-4">
        <SkeletonText lines={6} />
      </div>
    );
  }

  if (jobQuery.isError || !jobQuery.data) {
    return <ErrorState title="Job not found" description="This listing may have been removed." />;
  }

  const job = jobQuery.data;
  const salary = formatSalaryRange(job.salary_min, job.salary_max, job.salary_currency, job.salary_period);
  const applyUrl = job.source_url || null;
  const isSaved = job.is_saved;
  const isSaving = saveMutation.isPending || unsaveMutation.isPending;
  const matchScore = job.match?.match_score ?? job.match_score ?? job.relevance_score;

  return (
    <motion.div variants={fadeInUp} initial="hidden" animate="visible" className="mx-auto max-w-3xl space-y-4">
      <Card>
        <CardContent className="p-6 sm:p-8">
          <div className="flex items-start gap-4">
            <div className="flex size-14 shrink-0 items-center justify-center overflow-hidden rounded-[var(--radius-md)] bg-surface-2 text-ink-muted">
              {job.company.logo_url ? (
                <img src={job.company.logo_url} alt="" className="size-full object-cover" />
              ) : (
                <Building2 className="size-6" aria-hidden />
              )}
            </div>
            <div className="min-w-0 flex-1">
              <h1 className="text-xl font-semibold text-ink">{job.title}</h1>
              <Link to={paths.companyDetail(job.company.id)} className="text-sm font-medium text-brand hover:underline">
                {job.company.name}
              </Link>
              {job.company.industry && <p className="text-xs text-ink-muted">{job.company.industry}</p>}
            </div>
            {typeof matchScore === "number" && matchScore > 0 && (
              <button type="button" onClick={() => setPanelOpen(true)} aria-label="Why this score?" className="shrink-0">
                <ScoreRing value={Math.round(matchScore)} size={56} strokeWidth={5} label="match" />
              </button>
            )}
            <button
              type="button"
              onClick={() => (isSaved ? unsaveMutation.mutate(job.id) : saveMutation.mutate(job.id))}
              disabled={isSaving}
              className={cn(
                "flex size-9 shrink-0 items-center justify-center rounded-[var(--radius-md)] border border-border text-ink-muted transition-colors hover:bg-surface-2 disabled:cursor-wait",
                isSaved && "border-brand-border bg-brand-soft text-brand",
              )}
              aria-label={isSaved ? "Unsave job" : "Save job"}
            >
              {isSaved ? <BookmarkCheck className="size-4" aria-hidden /> : <Bookmark className="size-4" aria-hidden />}
            </button>
          </div>

          <div className="mt-4 flex flex-wrap gap-x-4 gap-y-2 text-sm text-ink-secondary">
            <span className="flex items-center gap-1.5">
              <MapPin className="size-3.5" aria-hidden />
              {formatLocation(job.location)}
            </span>
            <span className="flex items-center gap-1.5">
              <Briefcase className="size-3.5" aria-hidden />
              {formatEnumLabel(job.work_mode)} · {formatEnumLabel(job.employment_type)}
            </span>
            <span className="flex items-center gap-1.5">
              <Calendar className="size-3.5" aria-hidden />
              Posted {formatRelativeDate(job.posted_at)}
            </span>
            {salary && (
              <span className="flex items-center gap-1.5 font-medium text-ink">
                <Banknote className="size-3.5 text-success" aria-hidden />
                {salary}
              </span>
            )}
          </div>

          {salaryInsight && <p className="mt-2 text-xs text-ink-muted">{salaryInsight}</p>}

          {job.skills.length > 0 && (
            <div className="mt-4 flex flex-wrap gap-1.5">
              {job.skills.map((skill) => (
                <Badge key={skill.id} tone="neutral">
                  {skill.name}
                </Badge>
              ))}
            </div>
          )}

          <div className="mt-6 flex flex-wrap gap-2 border-y border-border py-4">
            {applyUrl ? (
              <a href={applyUrl} target="_blank" rel="noreferrer">
                <Button variant="gradient">
                  Apply on company site
                  <ExternalLink className="size-4" aria-hidden />
                </Button>
              </a>
            ) : (
              <Button variant="gradient" disabled>
                Apply on company site
                <ExternalLink className="size-4" aria-hidden />
              </Button>
            )}
            <Button variant="secondary" onClick={() => setTrackDialogOpen(true)}>
              <Send className="size-4" aria-hidden />
              Track this application
            </Button>
            <AutoApplyDialog
              jobId={job.id}
              jobTitle={job.title}
              companyName={job.company.name}
              jobDescription={job.description}
            />
            <Link to={`${paths.atsGenerator}?target_job_id=${encodeURIComponent(job.id)}&job_title=${encodeURIComponent(job.title)}&company_name=${encodeURIComponent(job.company.name)}`}>
              <Button variant="outline">
                <FileStack className="size-4" aria-hidden />
                Generate ATS Resume
              </Button>
            </Link>
            <Button variant="outline" onClick={() => askAiMutation.mutate()} isLoading={askAiMutation.isPending}>
              <MessageCircleQuestion className="size-4" aria-hidden />
              Ask AI Coach
            </Button>
          </div>

          {/* Deep links into the existing AI Career Coach tools, pre-filled
              with this job's title (and country, for salary) via router
              state — `CoachToolPage` reads `location.state` on mount. No new
              coach logic: same `/career-coach/*` endpoints the Coach page
              already calls. */}
          <div className="flex flex-wrap gap-2 border-b border-border py-4">
            <Link to={paths.coachTool("interview-prep")} state={{ targetRole: job.title }}>
              <Button variant="ghost" size="sm">
                <ListChecks className="size-3.5" aria-hidden />
                Prepare interview questions
              </Button>
            </Link>
            <Link to={paths.coachTool("roadmap")} state={{ targetRole: job.title }}>
              <Button variant="ghost" size="sm">
                <Compass className="size-3.5" aria-hidden />
                Generate study roadmap
              </Button>
            </Link>
            <Link
              to={paths.coachTool("salary-prediction")}
              state={{ targetRole: job.title, country: job.location?.country ?? undefined }}
            >
              <Button variant="ghost" size="sm">
                <DollarSign className="size-3.5" aria-hidden />
                Estimate salary
              </Button>
            </Link>
          </div>

          {job.ai_signals && (
            <div className="flex flex-wrap items-center gap-2 border-b border-border py-4 text-sm">
              <Badge tone={DIFFICULTY_TONE[job.ai_signals.application_difficulty]}>
                Interview difficulty (estimated): {formatEnumLabel(job.ai_signals.application_difficulty)}
              </Badge>
              {job.ai_signals.expected_ats_score != null && (
                <Badge tone="info">Expected ATS score: {Math.round(job.ai_signals.expected_ats_score)}</Badge>
              )}
              <span className="flex items-center gap-1 text-ink-muted">
                <TrendingUp className="size-3.5" aria-hidden />
                {Math.round(job.ai_signals.hiring_probability)}% hiring probability
              </span>
            </div>
          )}

          {job.match && (job.match.strengths.length > 0 || job.match.weaknesses.length > 0) && (
            <div className="grid gap-4 py-4 sm:grid-cols-2">
              {job.match.strengths.length > 0 && (
                <div>
                  <h3 className="mb-1.5 flex items-center gap-1.5 text-sm font-semibold text-ink">
                    <Sparkles className="size-3.5 text-brand" aria-hidden />
                    AI summary — strengths
                  </h3>
                  <ul className="space-y-1 text-sm text-ink-secondary">
                    {job.match.strengths.map((s, i) => (
                      <li key={i} className="flex items-start gap-1.5">
                        <CheckCircle2 className="mt-0.5 size-3.5 shrink-0 text-success" aria-hidden />
                        {s}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {job.match.weaknesses.length > 0 && (
                <div>
                  <h3 className="mb-1.5 flex items-center gap-1.5 text-sm font-semibold text-ink">
                    <AlertTriangle className="size-3.5 text-warning" aria-hidden />
                    AI summary — watch-outs
                  </h3>
                  <ul className="space-y-1 text-sm text-ink-secondary">
                    {job.match.weaknesses.map((w, i) => (
                      <li key={i} className="flex items-start gap-1.5">
                        <XCircle className="mt-0.5 size-3.5 shrink-0 text-danger" aria-hidden />
                        {w}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {job.match && job.match.missing_skills.length > 0 && (
            <div className="border-t border-border py-4">
              <h3 className="mb-1.5 text-sm font-semibold text-ink">Missing skills</h3>
              <div className="flex flex-wrap gap-1.5">
                {job.match.missing_skills.map((skill) => (
                  <Badge key={skill} tone="danger">
                    {skill}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          <div className="prose prose-sm mt-6 max-w-none space-y-4 text-sm leading-relaxed text-ink-secondary">
            {sections?.responsibilities && (
              <div>
                <h3 className="mb-1.5 text-sm font-semibold text-ink">Responsibilities</h3>
                <p className="whitespace-pre-wrap">{sections.responsibilities}</p>
              </div>
            )}
            {sections?.requirements && (
              <div>
                <h3 className="mb-1.5 text-sm font-semibold text-ink">Requirements</h3>
                <p className="whitespace-pre-wrap">{sections.requirements}</p>
              </div>
            )}
            {!sections?.responsibilities && !sections?.requirements && (
              <div>
                <h3 className="mb-1.5 text-sm font-semibold text-ink">Full description</h3>
                <p className="whitespace-pre-wrap">{sections?.rest ?? job.description}</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <Dialog
        open={trackDialogOpen}
        onOpenChange={setTrackDialogOpen}
        title="Track this application"
        description={`Add "${job.title}" at ${job.company.name} to your tracker as Applied.`}
        footer={
          <>
            <Button variant="ghost" onClick={() => setTrackDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={() => trackMutation.mutate()} isLoading={trackMutation.isPending}>
              Add to tracker
            </Button>
          </>
        }
      />

      <MatchScorePanel jobId={job.id} jobTitle={job.title} open={panelOpen} onOpenChange={setPanelOpen} />
    </motion.div>
  );
}
