import { motion } from "framer-motion";
import { History } from "lucide-react";
import { Link } from "react-router-dom";

import { staggerChildren } from "@/components/motion/variants";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { SkeletonText } from "@/components/ui/Skeleton";
import { VersionStatusBadge } from "@/features/resume-parser/components/VersionStatusBadge";
import type { ResumeVersionSummary } from "@/features/resume-parser/types";
import { formatRelativeDate } from "@/lib/format";
import { paths } from "@/routes/paths";

interface ResumeVersionsPanelProps {
  resumeId: string | null;
  versions: ResumeVersionSummary[];
  isLoading: boolean;
}

export function ResumeVersionsPanel({ resumeId, versions, isLoading }: ResumeVersionsPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-1.5">
          <History className="size-4 text-brand" aria-hidden />
          Resume versions
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <SkeletonText lines={3} />
        ) : versions.length === 0 ? (
          <EmptyState icon={History} title="No versions yet" description="Uploaded resume versions will appear here." />
        ) : (
          <motion.ul variants={staggerChildren(0.05)} initial="hidden" animate="visible" className="space-y-1">
            {versions.map((version) => (
              <motion.li key={version.id} variants={{ hidden: { opacity: 0 }, visible: { opacity: 1 } }}>
                <Link
                  to={resumeId ? paths.resumeVersion(resumeId, version.id) : paths.resumes}
                  className="flex items-center gap-3 rounded-[var(--radius-md)] px-2 py-2 transition-colors hover:bg-surface-2"
                >
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-ink">
                      Version {version.version_number}
                      {version.is_current && <span className="ml-1.5 text-xs text-brand">(current)</span>}
                    </p>
                    <p className="truncate text-xs text-ink-muted">{version.original_filename}</p>
                  </div>
                  <VersionStatusBadge status={version.status} />
                  <span className="w-16 shrink-0 text-right text-xs text-ink-muted">
                    {formatRelativeDate(version.created_at)}
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
