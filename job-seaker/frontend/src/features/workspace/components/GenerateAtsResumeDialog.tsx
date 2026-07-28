import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Download, Sparkles } from "lucide-react";
import { useState } from "react";

import { Badge } from "@/components/ui/Badge";
import { Button, type ButtonProps } from "@/components/ui/Button";
import { Card, CardContent } from "@/components/ui/Card";
import { Dialog } from "@/components/ui/Dialog";
import { ScoreRing } from "@/components/ui/ScoreRing";
import { Textarea } from "@/components/ui/Textarea";
import {
  createATSResume,
  getATSResumeDownloadUrl,
  getATSResumeVersion,
} from "@/features/ats-resume-generator/api";
import type { ATSFileFormat, ATSVersionStatus } from "@/features/ats-resume-generator/types";
import { useAuth } from "@/features/auth";
import { errorMessage } from "@/lib/errorMessage";
import { formatEnumLabel } from "@/lib/format";
import { toast } from "@/stores/toastStore";

const POLL_STATUSES: ATSVersionStatus[] = ["pending", "optimizing", "generating_files"];

interface GenerateAtsResumeDialogProps extends Omit<ButtonProps, "onClick"> {
  resumeVersionId: string;
  label?: string;
}

/**
 * Runs the ATS resume optimization for the current resume right from the
 * card that suggested it, instead of sending the user to a separate page.
 */
export function GenerateAtsResumeDialog({
  resumeVersionId,
  label = "Generate ATS resume",
  ...buttonProps
}: GenerateAtsResumeDialogProps) {
  const { user } = useAuth();
  const userId = user!.id;
  const queryClient = useQueryClient();

  const [open, setOpen] = useState(false);
  const [jobDescription, setJobDescription] = useState("");
  const [atsResumeId, setAtsResumeId] = useState<string | null>(null);
  const [versionId, setVersionId] = useState<string | null>(null);

  const create = useMutation({
    mutationFn: () =>
      createATSResume(userId, {
        resume_version_id: resumeVersionId,
        job_description: jobDescription,
      }),
    onSuccess: (resume) => {
      setAtsResumeId(resume.id);
      setVersionId(resume.current_version?.id ?? null);
      toast.success("Optimizing your resume", "This usually takes under a minute.");
    },
    onError: (error) =>
      toast.danger("Couldn't start optimization", errorMessage(error, "Check the job description and try again.")),
  });

  const versionQuery = useQuery({
    queryKey: ["ats-resume-generator", "version", atsResumeId, versionId],
    queryFn: () => getATSResumeVersion(userId, atsResumeId as string, versionId as string),
    enabled: !!atsResumeId && !!versionId,
    refetchInterval: (query) => (query.state.data && POLL_STATUSES.includes(query.state.data.status) ? 3000 : false),
  });

  const version = versionQuery.data;
  const isPolling = version && POLL_STATUSES.includes(version.status);

  async function handleDownload(format: ATSFileFormat) {
    if (!atsResumeId || !versionId) return;
    const url = await getATSResumeDownloadUrl(userId, atsResumeId, versionId, format);
    window.open(url, "_blank", "noopener,noreferrer");
  }

  function handleOpenChange(next: boolean) {
    setOpen(next);
    if (!next) {
      setJobDescription("");
      setAtsResumeId(null);
      setVersionId(null);
      if (version?.status === "completed") {
        void queryClient.invalidateQueries({ queryKey: ["workspace"] });
      }
    }
  }

  return (
    <>
      <Button variant="secondary" size="sm" onClick={() => setOpen(true)} {...buttonProps}>
        <Sparkles className="size-3.5" aria-hidden />
        {label}
      </Button>

      <Dialog
        open={open}
        onOpenChange={handleOpenChange}
        title="Generate a tailored ATS resume"
        description="Paste the target job description and we'll optimize your resume against it."
        footer={
          !atsResumeId ? (
            <>
              <Button variant="ghost" onClick={() => handleOpenChange(false)}>
                Cancel
              </Button>
              <Button
                onClick={() => create.mutate()}
                isLoading={create.isPending}
                disabled={jobDescription.trim().length < 20}
              >
                Optimize resume
              </Button>
            </>
          ) : (
            <Button variant="ghost" onClick={() => handleOpenChange(false)}>
              Close
            </Button>
          )
        }
      >
        {!atsResumeId ? (
          <Textarea
            placeholder="Paste the target job description here..."
            rows={7}
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
          />
        ) : (
          <div className="space-y-4">
            {isPolling && (
              <Card className="border-brand-border bg-brand-soft">
                <CardContent className="flex items-center gap-3 p-4">
                  <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1.2, ease: "linear" }}>
                    <Sparkles className="size-4 text-brand" aria-hidden />
                  </motion.div>
                  <p className="text-sm text-ink">
                    Claude is optimizing your resume — {formatEnumLabel(version!.status)}...
                  </p>
                </CardContent>
              </Card>
            )}

            {version?.status === "completed" && (
              <div className="space-y-3">
                <div className="flex items-center gap-6">
                  <div className="flex flex-col items-center gap-1.5">
                    <ScoreRing value={version.ats_score ?? 0} size={56} strokeWidth={5} />
                    <p className="text-xs text-ink-secondary">ATS score</p>
                  </div>
                  <div className="flex flex-col items-center gap-1.5">
                    <ScoreRing value={version.keyword_score ?? 0} size={56} strokeWidth={5} />
                    <p className="text-xs text-ink-secondary">Keyword score</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {(["docx", "pdf"] as ATSFileFormat[]).map((format) => (
                    <Button
                      key={format}
                      size="sm"
                      variant="outline"
                      disabled={!version.available_formats.includes(format)}
                      onClick={() => handleDownload(format)}
                    >
                      <Download className="size-3.5" aria-hidden />
                      {format.toUpperCase()}
                    </Button>
                  ))}
                </div>
              </div>
            )}

            {version?.status === "failed" && (
              <div>
                <Badge tone="danger">Optimization failed</Badge>
                <p className="mt-2 text-sm text-ink-secondary">{version.error_message ?? "Try again."}</p>
              </div>
            )}
          </div>
        )}
      </Dialog>
    </>
  );
}
