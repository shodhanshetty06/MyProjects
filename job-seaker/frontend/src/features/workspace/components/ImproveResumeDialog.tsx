import { useMutation } from "@tanstack/react-query";
import { FileSearch } from "lucide-react";
import { useEffect, useState } from "react";

import { Button, type ButtonProps } from "@/components/ui/Button";
import { Dialog } from "@/components/ui/Dialog";
import { EmptyState } from "@/components/ui/EmptyState";
import { reviewResume } from "@/features/ai-career-coach/api";
import { ResumeReviewView } from "@/features/ai-career-coach/components/ToolResults";
import { ThinkingIndicator } from "@/features/ai-career-coach/components/ThinkingIndicator";
import { useAuth } from "@/features/auth";

interface ImproveResumeDialogProps extends Omit<ButtonProps, "onClick"> {
  resumeVersionId: string;
  label?: string;
}

/**
 * Runs a structured resume critique inline from the workspace, instead of
 * sending the user to the coach tools page to get feedback.
 */
export function ImproveResumeDialog({
  resumeVersionId,
  label = "Improve resume",
  ...buttonProps
}: ImproveResumeDialogProps) {
  const { user } = useAuth();
  const userId = user!.id;

  const [open, setOpen] = useState(false);

  const review = useMutation({
    mutationFn: () => reviewResume(userId, resumeVersionId),
  });

  useEffect(() => {
    if (open && !review.data && !review.isPending) {
      review.mutate();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  function handleOpenChange(next: boolean) {
    setOpen(next);
    if (!next) review.reset();
  }

  return (
    <>
      <Button variant="secondary" size="sm" onClick={() => setOpen(true)} {...buttonProps}>
        <FileSearch className="size-3.5" aria-hidden />
        {label}
      </Button>

      <Dialog
        open={open}
        onOpenChange={handleOpenChange}
        title="Resume review"
        description="A structured critique of your current resume."
        footer={
          <Button variant="ghost" onClick={() => handleOpenChange(false)}>
            Close
          </Button>
        }
      >
        <div className="max-h-[60vh] overflow-y-auto pr-1">
          {review.isPending && <ThinkingIndicator label="Reviewing your resume..." />}
          {review.isError && (
            <EmptyState title="Something went wrong" description="Try running the review again." />
          )}
          {review.data && <ResumeReviewView result={review.data} />}
        </div>
      </Dialog>
    </>
  );
}
