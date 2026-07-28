import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Briefcase, Building2, ExternalLink, Pin, PinOff, Star, TrendingUp } from "lucide-react";
import { useParams } from "react-router-dom";

import { fadeInUp, staggerChildren } from "@/components/motion/variants";
import { Card, CardContent } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { ErrorState } from "@/components/ui/ErrorState";
import { SkeletonText } from "@/components/ui/Skeleton";
import { getCompany, listJobs } from "@/features/job-ingestion/api";
import { JobCard } from "@/features/search/components/JobCard";
import type { SearchResultItem } from "@/features/search/types";
import { cn } from "@/lib/cn";
import { usePinsStore } from "@/stores/pinsStore";

export function CompanyDetailPage() {
  const { companyId } = useParams<{ companyId: string }>();

  const companyQuery = useQuery({
    queryKey: ["company", companyId],
    queryFn: () => getCompany(companyId as string),
    enabled: !!companyId,
  });

  const jobsQuery = useQuery({
    queryKey: ["company-jobs", companyId],
    queryFn: () => listJobs({ company_id: companyId, page_size: 50 }),
    enabled: !!companyId,
  });

  const isPinned = usePinsStore((state) => (companyId ? state.isPinned("company", companyId) : false));
  const togglePin = usePinsStore((state) => state.togglePin);

  if (companyQuery.isLoading) {
    return (
      <div className="mx-auto max-w-3xl">
        <SkeletonText lines={5} />
      </div>
    );
  }

  if (companyQuery.isError || !companyQuery.data) {
    return <ErrorState title="Company not found" description="This company profile may have been removed." />;
  }

  const company = companyQuery.data;
  const intel = company.intelligence;

  // Adapts job_ingestion's JobRead shape to the SearchResultItem shape JobCard renders.
  const jobsAsSearchItems: SearchResultItem[] = (jobsQuery.data?.data ?? []).map((job) => ({
    id: job.id,
    title: job.title,
    company: { id: company.id, name: company.name, logo_url: company.logo_url, industry: company.industry },
    location: job.location,
    work_mode: job.work_mode,
    employment_type: job.employment_type,
    salary_min: job.salary_min,
    salary_max: job.salary_max,
    salary_currency: job.salary_currency,
    salary_period: job.salary_period,
    posted_at: job.posted_at,
    skills: job.skills,
    experience_level: null,
    relevance_score: null,
    match_score: null,
    keyword_score: null,
    quality_score: job.quality_score,
    quality_reasons: job.quality_reasons,
    seniority: job.seniority,
    tech_stack: job.tech_stack,
  }));

  return (
    <motion.div variants={fadeInUp} initial="hidden" animate="visible" className="mx-auto max-w-3xl space-y-6">
      <Card>
        <CardContent className="p-6 sm:p-8">
          <div className="flex items-start gap-4">
            <div className="flex size-14 shrink-0 items-center justify-center overflow-hidden rounded-[var(--radius-md)] bg-surface-2 text-ink-muted">
              {company.logo_url ? (
                <img src={company.logo_url} alt="" className="size-full object-cover" />
              ) : (
                <Building2 className="size-6" aria-hidden />
              )}
            </div>
            <div className="min-w-0 flex-1">
              <h1 className="text-xl font-semibold text-ink">{company.name}</h1>
              {company.industry && <p className="text-sm text-ink-secondary">{company.industry}</p>}
            </div>
            <button
              type="button"
              onClick={() => togglePin({ id: company.id, type: "company", label: company.name, meta: company.industry ?? undefined })}
              className={cn(
                "flex size-9 shrink-0 items-center justify-center rounded-[var(--radius-md)] border border-border text-ink-muted transition-colors hover:bg-surface-2",
                isPinned && "border-brand-border bg-brand-soft text-brand",
              )}
              aria-label={isPinned ? "Unpin company" : "Pin company"}
            >
              {isPinned ? <PinOff className="size-4" aria-hidden /> : <Pin className="size-4" aria-hidden />}
            </button>
          </div>

          {company.description && <p className="mt-4 text-sm leading-relaxed text-ink-secondary">{company.description}</p>}

          <div className="mt-4 flex flex-wrap gap-4 text-sm text-ink-secondary">
            {company.website && (
              <a href={company.website} target="_blank" rel="noreferrer" className="flex items-center gap-1.5 text-brand hover:underline">
                <ExternalLink className="size-3.5" aria-hidden />
                Website
              </a>
            )}
            {intel?.company_rating != null && (
              <span className="flex items-center gap-1.5">
                <Star className="size-3.5 text-warning" aria-hidden />
                {intel.company_rating.toFixed(1)} rating
              </span>
            )}
            {intel?.hiring_frequency_per_month != null && (
              <span className="flex items-center gap-1.5">
                <TrendingUp className="size-3.5" aria-hidden />
                ~{Math.round(intel.hiring_frequency_per_month)} hires/mo
              </span>
            )}
            {intel?.company_size && <span>{intel.company_size} employees</span>}
            {intel?.funding_stage && <span>{intel.funding_stage}</span>}
            {intel?.remote_policy && intel.remote_policy !== "unknown" && (
              <span className="capitalize">{intel.remote_policy.replace("_", " ")}</span>
            )}
            {company.avg_salary_yearly != null && (
              <span>
                ~{Math.round(company.avg_salary_yearly).toLocaleString()} {company.avg_salary_currency ?? ""} avg
                salary
              </span>
            )}
          </div>

          {intel?.tech_stack && intel.tech_stack.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {intel.tech_stack.map((tech) => (
                <span
                  key={tech}
                  className="rounded-full border border-border bg-surface-2 px-2 py-0.5 text-xs text-ink-secondary"
                >
                  {tech}
                </span>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <div>
        <h2 className="mb-3 flex items-center gap-1.5 text-sm font-semibold text-ink">
          <Briefcase className="size-4 text-brand" aria-hidden />
          Open roles ({jobsQuery.data?.pagination.total_items ?? 0})
        </h2>
        {jobsQuery.isLoading ? (
          <SkeletonText lines={4} />
        ) : jobsAsSearchItems.length === 0 ? (
          <EmptyState icon={Briefcase} title="No open roles listed" description="Check back later for new postings." />
        ) : (
          <motion.div variants={staggerChildren(0.05)} initial="hidden" animate="visible" className="space-y-3">
            {jobsAsSearchItems.map((job) => (
              <JobCard key={job.id} job={job} />
            ))}
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}
