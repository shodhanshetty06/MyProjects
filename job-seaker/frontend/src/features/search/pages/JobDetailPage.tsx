import { useMutation, useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import {
  Banknote,
  Briefcase,
  Building2,
  Calendar,
  ExternalLink,
  MapPin,
  Pin,
  PinOff,
  Send,
} from "lucide-react";
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { fadeInUp } from "@/components/motion/variants";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent } from "@/components/ui/Card";
import { Dialog } from "@/components/ui/Dialog";
import { ErrorState } from "@/components/ui/ErrorState";
import { SkeletonText } from "@/components/ui/Skeleton";
import { useAuth } from "@/features/auth";
import { getCompany, getJob } from "@/features/job-ingestion/api";
import { createApplication } from "@/features/tracking/api";
import { cn } from "@/lib/cn";
import { formatEnumLabel, formatLocation, formatRelativeDate, formatSalaryRange } from "@/lib/format";
import { paths } from "@/routes/paths";
import { usePinsStore } from "@/stores/pinsStore";
import { toast } from "@/stores/toastStore";

export function JobDetailPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const { user } = useAuth();
  const navigate = useNavigate();
  const [trackDialogOpen, setTrackDialogOpen] = useState(false);

  const jobQuery = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId as string),
    enabled: !!jobId,
  });

  const companyQuery = useQuery({
    queryKey: ["company", jobQuery.data?.company_id],
    queryFn: () => getCompany(jobQuery.data!.company_id),
    enabled: !!jobQuery.data?.company_id,
  });

  const isPinned = usePinsStore((state) => (jobId ? state.isPinned("job", jobId) : false));
  const togglePin = usePinsStore((state) => state.togglePin);

  const trackMutation = useMutation({
    mutationFn: () =>
      createApplication(user!.id, {
        job_id: jobQuery.data!.id,
        company_name: companyQuery.data?.name ?? "Unknown company",
        job_title: jobQuery.data!.title,
        job_url: jobQuery.data!.source_url,
        location: jobQuery.data?.location?.raw_text ?? null,
        source: "manual",
      }),
    onSuccess: (app) => {
      toast.success("Application tracked", `${app.job_title} added to your tracker.`);
      setTrackDialogOpen(false);
      navigate(paths.trackerTimeline(app.id));
    },
    onError: () => toast.danger("Couldn't track this application", "Try again in a moment."),
  });

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

  return (
    <motion.div variants={fadeInUp} initial="hidden" animate="visible" className="mx-auto max-w-3xl">
      <Card>
        <CardContent className="p-6 sm:p-8">
          <div className="flex items-start gap-4">
            <div className="flex size-14 shrink-0 items-center justify-center overflow-hidden rounded-[var(--radius-md)] bg-surface-2 text-ink-muted">
              {companyQuery.data?.logo_url ? (
                <img src={companyQuery.data.logo_url} alt="" className="size-full object-cover" />
              ) : (
                <Building2 className="size-6" aria-hidden />
              )}
            </div>
            <div className="min-w-0 flex-1">
              <h1 className="text-xl font-semibold text-ink">{job.title}</h1>
              {companyQuery.data ? (
                <Link
                  to={paths.companyDetail(companyQuery.data.id)}
                  className="text-sm font-medium text-brand hover:underline"
                >
                  {companyQuery.data.name}
                </Link>
              ) : (
                <p className="text-sm text-ink-secondary">Loading company...</p>
              )}
            </div>
            <button
              type="button"
              onClick={() =>
                togglePin({ id: job.id, type: "job", label: job.title, meta: companyQuery.data?.name })
              }
              className={cn(
                "flex size-9 shrink-0 items-center justify-center rounded-[var(--radius-md)] border border-border text-ink-muted transition-colors hover:bg-surface-2",
                isPinned && "border-brand-border bg-brand-soft text-brand",
              )}
              aria-label={isPinned ? "Unpin job" : "Pin job"}
            >
              {isPinned ? <PinOff className="size-4" aria-hidden /> : <Pin className="size-4" aria-hidden />}
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
            <a href={job.source_url} target="_blank" rel="noreferrer">
              <Button variant="gradient">
                Apply on company site
                <ExternalLink className="size-4" aria-hidden />
              </Button>
            </a>
            <Button variant="secondary" onClick={() => setTrackDialogOpen(true)}>
              Track this application
            </Button>
            <Link to={paths.autoApply}>
              <Button variant="outline">
                <Send className="size-4" aria-hidden />
                Queue auto-apply
              </Button>
            </Link>
          </div>

          <div className="prose prose-sm mt-6 max-w-none whitespace-pre-wrap text-sm leading-relaxed text-ink-secondary">
            {job.description}
          </div>
        </CardContent>
      </Card>

      <Dialog
        open={trackDialogOpen}
        onOpenChange={setTrackDialogOpen}
        title="Track this application"
        description={`Add "${job.title}" at ${companyQuery.data?.name ?? "this company"} to your tracker as Applied.`}
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
    </motion.div>
  );
}
