import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Search } from "lucide-react";
import { Link } from "react-router-dom";

import { PageHeader } from "@/components/layout/PageHeader";
import { staggerChildren } from "@/components/motion/variants";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { ErrorState } from "@/components/ui/ErrorState";
import { SkeletonCard } from "@/components/ui/Skeleton";
import { JobCard } from "@/features/search/components/JobCard";
import { listSavedJobs } from "@/features/search/api";
import { paths } from "@/routes/paths";

export function SavedJobsPage() {
  const savedJobsQuery = useQuery({
    queryKey: ["saved-jobs"],
    queryFn: () => listSavedJobs(1, 50),
  });

  const jobs = savedJobsQuery.data?.data ?? [];

  return (
    <div>
      <PageHeader
        eyebrow="Job search"
        title="Saved jobs"
        description="Jobs you've saved while browsing - synced to your account across devices."
      />

      {savedJobsQuery.isLoading ? (
        <div className="grid gap-3 sm:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      ) : savedJobsQuery.isError ? (
        <ErrorState
          description="Couldn't load your saved jobs. Try again."
          action={
            <button onClick={() => savedJobsQuery.refetch()} className="text-sm font-medium text-brand">
              Retry
            </button>
          }
        />
      ) : jobs.length === 0 ? (
        <EmptyState
          icon={Search}
          title="No saved jobs yet"
          description="Save jobs from search results or a job's detail page to keep them here."
          action={
            <Link to={paths.search}>
              <Button variant="secondary">Browse jobs</Button>
            </Link>
          }
        />
      ) : (
        <motion.div variants={staggerChildren(0.05)} initial="hidden" animate="visible" className="grid gap-3 sm:grid-cols-2">
          {jobs.map((job) => (
            <motion.div key={job.id} variants={{ hidden: { opacity: 0, y: 8 }, visible: { opacity: 1, y: 0 } }}>
              <JobCard job={job} />
            </motion.div>
          ))}
        </motion.div>
      )}
    </div>
  );
}
