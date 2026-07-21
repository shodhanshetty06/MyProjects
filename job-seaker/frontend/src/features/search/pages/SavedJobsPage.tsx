import { motion } from "framer-motion";
import { Building2, PinOff, Search } from "lucide-react";
import { useMemo } from "react";
import { Link } from "react-router-dom";

import { PageHeader } from "@/components/layout/PageHeader";
import { staggerChildren } from "@/components/motion/variants";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { paths } from "@/routes/paths";
import { usePinsStore } from "@/stores/pinsStore";

export function SavedJobsPage() {
  const allItems = usePinsStore((state) => state.items);
  const unpin = usePinsStore((state) => state.unpin);
  const jobs = useMemo(() => allItems.filter((item) => item.type === "job"), [allItems]);

  return (
    <div>
      <PageHeader
        eyebrow="Job search"
        title="Saved jobs"
        description="Jobs you've pinned while browsing - stored locally on this device."
      />

      {jobs.length === 0 ? (
        <EmptyState
          icon={Search}
          title="No saved jobs yet"
          description="Pin jobs from search results or a job's detail page to keep them here."
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
              <Card className="flex items-center gap-3 p-4">
                <div className="flex size-10 shrink-0 items-center justify-center rounded-[var(--radius-md)] bg-surface-2 text-ink-muted">
                  <Building2 className="size-4" aria-hidden />
                </div>
                <div className="min-w-0 flex-1">
                  <Link to={paths.jobDetail(job.id)} className="truncate text-sm font-medium text-ink hover:text-brand">
                    {job.label}
                  </Link>
                  {job.meta && <p className="truncate text-xs text-ink-muted">{job.meta}</p>}
                </div>
                <button
                  type="button"
                  onClick={() => unpin("job", job.id)}
                  className="shrink-0 rounded-[var(--radius-sm)] p-1.5 text-ink-muted transition-colors hover:bg-surface-2 hover:text-ink"
                  aria-label={`Unpin ${job.label}`}
                >
                  <PinOff className="size-4" aria-hidden />
                </button>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      )}
    </div>
  );
}
