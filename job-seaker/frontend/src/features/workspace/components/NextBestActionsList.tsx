import { motion } from "framer-motion";
import { ArrowRight, ListChecks } from "lucide-react";
import { Link } from "react-router-dom";

import { staggerChildren } from "@/components/motion/variants";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { SkeletonText } from "@/components/ui/Skeleton";
import { FixSkillGapsDialog } from "@/features/workspace/components/FixSkillGapsDialog";
import { GenerateAtsResumeDialog } from "@/features/workspace/components/GenerateAtsResumeDialog";
import { UploadResumeButton } from "@/features/workspace/components/UploadResumeButton";
import type { NextBestAction, ResumeHealth } from "@/features/workspace/types";

interface NextBestActionsListProps {
  actions: NextBestAction[];
  resumeHealth: ResumeHealth | undefined;
  isLoading: boolean;
}

const itemVariants = { hidden: { opacity: 0, x: -8 }, visible: { opacity: 1, x: 0 } };

export function NextBestActionsList({ actions, resumeHealth, isLoading }: NextBestActionsListProps) {
  const resumeVersionId = resumeHealth?.resume_version_id ?? null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-1.5">
          <ListChecks className="size-4 text-brand" aria-hidden />
          Next best actions
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <SkeletonText lines={3} />
        ) : (
          <motion.ul variants={staggerChildren(0.05)} initial="hidden" animate="visible" className="space-y-2">
            {actions.map((action) => (
              <motion.li
                key={action.key}
                variants={itemVariants}
                className="flex items-center gap-3 rounded-[var(--radius-md)] border border-border p-3"
              >
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium text-ink">{action.title}</p>
                  <p className="mt-0.5 line-clamp-2 text-xs text-ink-secondary">{action.description}</p>
                </div>
                {action.key === "upload_resume" ? (
                  <UploadResumeButton label="Upload" />
                ) : action.key === "generate_ats_resume" && resumeVersionId ? (
                  <GenerateAtsResumeDialog resumeVersionId={resumeVersionId} label="Generate" />
                ) : action.key === "close_skill_gaps" && resumeVersionId ? (
                  <FixSkillGapsDialog resumeVersionId={resumeVersionId} label="Fix skills" />
                ) : (
                  <Link
                    to={action.cta_path}
                    className="group flex shrink-0 items-center gap-1.5 rounded-[var(--radius-sm)] px-1 text-xs font-medium text-ink-secondary transition-colors hover:text-brand"
                  >
                    View
                    <ArrowRight
                      className="size-3.5 transition-transform group-hover:translate-x-0.5"
                      aria-hidden
                    />
                  </Link>
                )}
              </motion.li>
            ))}
          </motion.ul>
        )}
      </CardContent>
    </Card>
  );
}
