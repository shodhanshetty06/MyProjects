import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Briefcase, Calendar, Inbox, Mail, Plus } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { PageHeader } from "@/components/layout/PageHeader";
import { staggerChildren } from "@/components/motion/variants";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/DropdownMenu";
import { SkeletonCard } from "@/components/ui/Skeleton";
import { useAuth } from "@/features/auth";
import { changeApplicationStatus, getApplicationStatistics, listApplications } from "@/features/tracking/api";
import { ApplicationStatusBadge } from "@/features/tracking/components/ApplicationStatusBadge";
import { CreateApplicationDialog } from "@/features/tracking/components/CreateApplicationDialog";
import { APPLICATION_STATUSES, type ApplicationStatus } from "@/features/tracking/types";
import { formatEnumLabel, formatRelativeDate } from "@/lib/format";
import { paths } from "@/routes/paths";
import { toast } from "@/stores/toastStore";

export function TrackerBoardPage() {
  const { user } = useAuth();
  const userId = user!.id;
  const queryClient = useQueryClient();
  const [createOpen, setCreateOpen] = useState(false);

  const applicationsQuery = useQuery({
    queryKey: ["applications", userId, "board"],
    queryFn: () => listApplications(userId, { page_size: 100, sort_by: "applied_at", sort_direction: "desc" }),
  });

  const statisticsQuery = useQuery({
    queryKey: ["application-statistics", userId],
    queryFn: () => getApplicationStatistics(userId),
  });

  const statusMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: ApplicationStatus }) =>
      changeApplicationStatus(userId, id, { status }),
    onSuccess: () => {
      toast.success("Status updated");
      void queryClient.invalidateQueries({ queryKey: ["applications", userId] });
      void queryClient.invalidateQueries({ queryKey: ["application-statistics", userId] });
    },
  });

  const columns = useMemo(() => {
    const apps = applicationsQuery.data?.data ?? [];
    return APPLICATION_STATUSES.map((status) => ({
      status,
      items: apps.filter((a) => a.status === status),
    }));
  }, [applicationsQuery.data]);

  const stats = statisticsQuery.data;

  return (
    <div>
      <PageHeader
        eyebrow="Applications"
        title="Application tracker"
        description="Every application, from applied to offer, in one board."
        actions={
          <div className="flex items-center gap-2">
            <Link to={paths.trackerCalendar}>
              <Button variant="secondary">
                <Calendar className="size-4" aria-hidden />
                Calendar
              </Button>
            </Link>
            <Link to={paths.trackerEmails}>
              <Button variant="secondary">
                <Mail className="size-4" aria-hidden />
                Tracked emails
              </Button>
            </Link>
            <Button onClick={() => setCreateOpen(true)}>
              <Plus className="size-4" aria-hidden />
              Add application
            </Button>
          </div>
        }
      />

      {stats && (
        <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Card className="p-4">
            <p className="text-xs text-ink-secondary">Total applications</p>
            <p className="mt-1 text-xl font-semibold text-ink">{stats.total_applications}</p>
          </Card>
          <Card className="p-4">
            <p className="text-xs text-ink-secondary">This week</p>
            <p className="mt-1 text-xl font-semibold text-ink">{stats.applications_last_7_days}</p>
          </Card>
          <Card className="p-4">
            <p className="text-xs text-ink-secondary">Interview rate</p>
            <p className="mt-1 text-xl font-semibold text-ink">{Math.round(stats.interview_rate * 100)}%</p>
          </Card>
          <Card className="p-4">
            <p className="text-xs text-ink-secondary">Offer rate</p>
            <p className="mt-1 text-xl font-semibold text-ink">{Math.round(stats.offer_rate * 100)}%</p>
          </Card>
        </div>
      )}

      {applicationsQuery.isLoading ? (
        <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      ) : (applicationsQuery.data?.data.length ?? 0) === 0 ? (
        <div className="rounded-[var(--radius-lg)] border border-dashed border-border py-16 text-center">
          <Inbox className="mx-auto size-8 text-ink-muted" aria-hidden />
          <p className="mt-3 text-sm font-medium text-ink">No applications tracked yet</p>
          <p className="mt-1 text-sm text-ink-secondary">Add one manually, or track it from a job's detail page.</p>
          <Button className="mt-4" onClick={() => setCreateOpen(true)}>
            <Plus className="size-4" aria-hidden />
            Add application
          </Button>
        </div>
      ) : (
        <div className="flex gap-4 overflow-x-auto pb-4">
          {columns.map((column) => (
            <div key={column.status} className="w-72 shrink-0">
              <div className="mb-2 flex items-center justify-between px-1">
                <ApplicationStatusBadge status={column.status} />
                <span className="text-xs text-ink-muted">{column.items.length}</span>
              </div>
              <motion.div variants={staggerChildren(0.04)} initial="hidden" animate="visible" className="space-y-2">
                {column.items.map((app) => (
                  <motion.div key={app.id} variants={{ hidden: { opacity: 0, y: 6 }, visible: { opacity: 1, y: 0 } }}>
                    <Card className="p-3">
                      <Link to={paths.trackerTimeline(app.id)} className="block">
                        <p className="truncate text-sm font-medium text-ink hover:text-brand">{app.job_title}</p>
                        <p className="truncate text-xs text-ink-secondary">{app.company_name}</p>
                        <p className="mt-1 flex items-center gap-1 text-[11px] text-ink-muted">
                          <Briefcase className="size-3" aria-hidden />
                          {formatRelativeDate(app.applied_at)}
                        </p>
                      </Link>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <button className="mt-2 text-[11px] font-medium text-ink-muted hover:text-ink">
                            Move to...
                          </button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="start">
                          {APPLICATION_STATUSES.filter((s) => s !== app.status).map((s) => (
                            <DropdownMenuItem key={s} onSelect={() => statusMutation.mutate({ id: app.id, status: s })}>
                              {formatEnumLabel(s)}
                            </DropdownMenuItem>
                          ))}
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </Card>
                  </motion.div>
                ))}
              </motion.div>
            </div>
          ))}
        </div>
      )}

      <CreateApplicationDialog open={createOpen} onOpenChange={setCreateOpen} />
    </div>
  );
}
