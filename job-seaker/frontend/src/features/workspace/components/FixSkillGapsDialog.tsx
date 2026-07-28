import { useMutation } from "@tanstack/react-query";
import { ListChecks } from "lucide-react";
import { useState } from "react";

import { Button, type ButtonProps } from "@/components/ui/Button";
import { Dialog } from "@/components/ui/Dialog";
import { EmptyState } from "@/components/ui/EmptyState";
import { Input } from "@/components/ui/Input";
import { analyzeSkillGap } from "@/features/ai-career-coach/api";
import { SkillGapView } from "@/features/ai-career-coach/components/ToolResults";
import { ThinkingIndicator } from "@/features/ai-career-coach/components/ThinkingIndicator";
import { useAuth } from "@/features/auth";

interface FixSkillGapsDialogProps extends Omit<ButtonProps, "onClick"> {
  resumeVersionId: string;
  defaultTargetRole?: string;
  label?: string;
}

/**
 * Runs skill-gap analysis inline from the workspace card that flagged the
 * gap, instead of sending the user to the coach tools page to do it.
 */
export function FixSkillGapsDialog({
  resumeVersionId,
  defaultTargetRole = "",
  label = "Fix missing skills",
  ...buttonProps
}: FixSkillGapsDialogProps) {
  const { user } = useAuth();
  const userId = user!.id;

  const [open, setOpen] = useState(false);
  const [targetRole, setTargetRole] = useState(defaultTargetRole);

  const analyze = useMutation({
    mutationFn: () => analyzeSkillGap(userId, resumeVersionId, targetRole),
  });

  function handleOpenChange(next: boolean) {
    setOpen(next);
    if (!next) analyze.reset();
  }

  return (
    <>
      <Button variant="secondary" size="sm" onClick={() => setOpen(true)} {...buttonProps}>
        <ListChecks className="size-3.5" aria-hidden />
        {label}
      </Button>

      <Dialog
        open={open}
        onOpenChange={handleOpenChange}
        title="Close your skill gaps"
        description="Tell us the role you're targeting and we'll compare it against your resume."
        footer={
          !analyze.data ? (
            <>
              <Button variant="ghost" onClick={() => handleOpenChange(false)}>
                Cancel
              </Button>
              <Button
                onClick={() => analyze.mutate()}
                isLoading={analyze.isPending}
                disabled={targetRole.trim().length === 0}
              >
                Analyze
              </Button>
            </>
          ) : (
            <Button variant="ghost" onClick={() => handleOpenChange(false)}>
              Close
            </Button>
          )
        }
      >
        {!analyze.data ? (
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-ink-secondary">Target role</label>
            <Input
              value={targetRole}
              onChange={(e) => setTargetRole(e.target.value)}
              placeholder="e.g. Senior Frontend Engineer"
            />
            {analyze.isPending && <ThinkingIndicator label="Analyzing..." />}
            {analyze.isError && (
              <EmptyState title="Something went wrong" description="Try running the analysis again." />
            )}
          </div>
        ) : (
          <div className="max-h-[60vh] overflow-y-auto pr-1">
            <SkillGapView result={analyze.data} />
          </div>
        )}
      </Dialog>
    </>
  );
}
