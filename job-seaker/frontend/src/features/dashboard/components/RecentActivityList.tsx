import { motion } from "framer-motion";
import { Briefcase } from "lucide-react";
import { Link } from "react-router-dom";

import { staggerChildren } from "@/components/motion/variants";
import { Badge, type BadgeTone } from "@/components/ui/Badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { SkeletonText } from "@/components/ui/Skeleton";
import type { ApplicationRead, ApplicationStatus } from "@/features/tracking/types";
import { paths } from "@/routes/paths";

const STATUS_TONE: Record<ApplicationStatus, BadgeTone> = {
  applied: "info",
  assessment: "warning",
  interview: "brand",
  offer: "success",
  accepted: "success",
  rejected: "danger",
  withdrawn: "neutral",
};

function relativeTime(iso: string): string {
  const diffMs = Date.now() - new Date(iso).getTime();
  const days = Math.floor(diffMs / 86_400_000);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return `${days}d ago`;
  return `${Math.floor(days / 7)}w ago`;
}

interface RecentActivityListProps {
  applications: ApplicationRead[];
  isLoading: boolean;
}

export function RecentActivityList({ applications, isLoading }: RecentActivityListProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-1.5">
          <Briefcase className="size-4 text-brand" aria-hidden />
          Recent activity
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <SkeletonText lines={4} />
        ) : applications.length === 0 ? (
          <EmptyState
            icon={Briefcase}
            title="No applications yet"
            description="Track applications as you apply to see your activity here."
          />
        ) : (
          <motion.ul variants={staggerChildren(0.05)} initial="hidden" animate="visible" className="space-y-1">
            {applications.map((app) => (
              <motion.li key={app.id} variants={{ hidden: { opacity: 0 }, visible: { opacity: 1 } }}>
                <Link
                  to={paths.trackerTimeline(app.id)}
                  className="flex items-center gap-3 rounded-[var(--radius-md)] px-2 py-2 transition-colors hover:bg-surface-2"
                >
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-ink">{app.job_title}</p>
                    <p className="truncate text-xs text-ink-muted">{app.company_name}</p>
                  </div>
                  <Badge tone={STATUS_TONE[app.status]} className="capitalize">
                    {app.status}
                  </Badge>
                  <span className="w-12 shrink-0 text-right text-xs text-ink-muted">
                    {relativeTime(app.applied_at)}
                  </span>
                </Link>
              </motion.li>
            ))}
          </motion.ul>
        )}
      </CardContent>
    </Card>
  );
}
