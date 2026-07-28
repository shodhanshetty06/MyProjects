import { useMutation } from "@tanstack/react-query";
import { FileStack, Send, Sparkles, Upload } from "lucide-react";
import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { Button, type ButtonProps } from "@/components/ui/Button";
import { Dialog } from "@/components/ui/Dialog";
import { Select } from "@/components/ui/Select";
import { createATSResume, getATSResumeVersion } from "@/features/ats-resume-generator/api";
import type { ATSVersionStatus } from "@/features/ats-resume-generator/types";
import { createAutoApplyTask } from "@/features/auto-apply/api";
import { useAuth } from "@/features/auth";
import { uploadResume } from "@/features/resume-parser/api";
import { useParsedResumeOptions } from "@/features/resume-parser/hooks";
import { errorMessage } from "@/lib/errorMessage";
import { paths } from "@/routes/paths";
import { toast } from "@/stores/toastStore";

const POLL_STATUSES: ATSVersionStatus[] = ["pending", "optimizing", "generating_files"];
const MAX_POLL_ATTEMPTS = 40;
const POLL_INTERVAL_MS = 3000;

function sleep(ms: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, ms));
}

interface AutoApplyDialogProps extends Omit<ButtonProps, "onClick"> {
  jobId: string;
  jobTitle: string;
  companyName: string;
  jobDescription: string;
  label?: string;
}

/**
 * Entry point for Auto Apply from a job card/detail view. Clicking the
 * button opens a choice between two resume sources — tailor a fresh ATS
 * resume from this job's description, or reuse/upload a resume as-is —
 * either path ends the same way: an `auto-apply` task is created and the
 * user is dropped onto its detail/progress page.
 */
export function AutoApplyDialog({
  jobId,
  jobTitle,
  companyName,
  jobDescription,
  label = "Auto Apply",
  ...buttonProps
}: AutoApplyDialogProps) {
  const { user } = useAuth();
  const userId = user!.id;
  const navigate = useNavigate();

  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<"choice" | "pick-resume">("choice");
  const [selectedResumeVersionId, setSelectedResumeVersionId] = useState("");
  const [statusText, setStatusText] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const { options: resumeOptions, isLoading: resumesLoading } = useParsedResumeOptions(userId);

  function goToTask(taskId: string) {
    setOpen(false);
    navigate(paths.autoApplyTask(taskId));
  }

  const generateAndApply = useMutation({
    mutationFn: async () => {
      const sourceResumeVersionId = selectedResumeVersionId || resumeOptions[0]?.value;
      if (!sourceResumeVersionId) {
        throw new Error("Upload a resume first so we have something to tailor.");
      }
      if (jobDescription.trim().length < 20) {
        throw new Error("This job has no usable description to tailor a resume against.");
      }

      setStatusText("Tailoring a resume to this job...");
      const atsResume = await createATSResume(userId, {
        resume_version_id: sourceResumeVersionId,
        job_description: jobDescription,
        job_title: jobTitle,
        company_name: companyName,
        target_job_id: jobId,
      });
      const versionId = atsResume.current_version?.id;
      if (!versionId) throw new Error("Resume generation didn't start. Try again.");

      let version = await getATSResumeVersion(userId, atsResume.id, versionId);
      let attempts = 0;
      while (POLL_STATUSES.includes(version.status) && attempts < MAX_POLL_ATTEMPTS) {
        await sleep(POLL_INTERVAL_MS);
        version = await getATSResumeVersion(userId, atsResume.id, versionId);
        attempts += 1;
      }
      if (version.status !== "completed") {
        throw new Error(version.error_message ?? "Resume generation timed out. Try again.");
      }

      setStatusText("Submitting your application...");
      return createAutoApplyTask(userId, { job_id: jobId, ats_resume_id: atsResume.id });
    },
    onSuccess: (task) => {
      toast.success("Auto Apply started", `Applying to ${jobTitle} at ${companyName} with your tailored resume.`);
      goToTask(task.id);
    },
    onError: (error) => {
      toast.danger("Couldn't start Auto Apply", errorMessage(error, "Try again in a moment."));
      setStatusText("");
    },
  });

  const applyWithResume = useMutation({
    mutationFn: (resumeVersionId: string) =>
      createAutoApplyTask(userId, { job_id: jobId, resume_version_id: resumeVersionId }),
    onSuccess: (task) => {
      toast.success("Auto Apply started", `Applying to ${jobTitle} at ${companyName}.`);
      goToTask(task.id);
    },
    onError: (error) => toast.danger("Couldn't start Auto Apply", errorMessage(error, "Try again in a moment.")),
  });

  const uploadAndApply = useMutation({
    mutationFn: async (file: File) => {
      const resume = await uploadResume(userId, file);
      const versionId = resume.current_version?.id;
      if (!versionId) throw new Error("Upload didn't finish processing. Try again.");
      return createAutoApplyTask(userId, { job_id: jobId, resume_version_id: versionId });
    },
    onSuccess: (task) => {
      toast.success("Auto Apply started", `Applying to ${jobTitle} at ${companyName}.`);
      goToTask(task.id);
    },
    onError: (error) => toast.danger("Couldn't start Auto Apply", errorMessage(error, "Try again in a moment.")),
  });

  const isBusy = generateAndApply.isPending || applyWithResume.isPending || uploadAndApply.isPending;

  function handleOpenChange(next: boolean) {
    if (isBusy) return;
    setOpen(next);
    if (!next) {
      setMode("choice");
      setStatusText("");
      setSelectedResumeVersionId("");
    }
  }

  return (
    <>
      <Button variant="gradient" onClick={() => setOpen(true)} {...buttonProps}>
        <Send className="size-4" aria-hidden />
        {label}
      </Button>

      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.doc,.docx"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (file) uploadAndApply.mutate(file);
        }}
      />

      <Dialog
        open={open}
        onOpenChange={handleOpenChange}
        title="Auto Apply"
        description={`Apply to "${jobTitle}" at ${companyName} automatically.`}
      >
        {mode === "choice" && (
          <div className="space-y-2">
            <button
              type="button"
              disabled={isBusy}
              onClick={() => generateAndApply.mutate()}
              className="flex w-full items-start gap-3 rounded-[var(--radius-md)] border border-border p-3 text-left transition-colors hover:border-brand-border hover:bg-brand-soft disabled:cursor-wait disabled:opacity-60"
            >
              <Sparkles className="mt-0.5 size-4 shrink-0 text-brand" aria-hidden />
              <span>
                <span className="block text-sm font-medium text-ink">
                  {generateAndApply.isPending ? statusText || "Working..." : "Generate ATS resume & apply"}
                </span>
                <span className="block text-xs text-ink-secondary">
                  Tailor a resume to this job's description with AI, then submit automatically.
                </span>
              </span>
            </button>

            <button
              type="button"
              disabled={isBusy}
              onClick={() => setMode("pick-resume")}
              className="flex w-full items-start gap-3 rounded-[var(--radius-md)] border border-border p-3 text-left transition-colors hover:border-brand-border hover:bg-brand-soft disabled:cursor-wait disabled:opacity-60"
            >
              <FileStack className="mt-0.5 size-4 shrink-0 text-brand" aria-hidden />
              <span>
                <span className="block text-sm font-medium text-ink">Apply with an uploaded resume</span>
                <span className="block text-xs text-ink-secondary">
                  Use a resume you've already uploaded, or upload a new one now.
                </span>
              </span>
            </button>
          </div>
        )}

        {mode === "pick-resume" && (
          <div className="space-y-3">
            {resumeOptions.length > 0 && (
              <>
                <Select
                  value={selectedResumeVersionId}
                  onValueChange={setSelectedResumeVersionId}
                  options={resumeOptions}
                  placeholder="Choose a resume"
                  disabled={isBusy}
                />
                <Button
                  className="w-full"
                  disabled={!selectedResumeVersionId || isBusy}
                  isLoading={applyWithResume.isPending}
                  onClick={() => applyWithResume.mutate(selectedResumeVersionId)}
                >
                  Apply with this resume
                </Button>
              </>
            )}

            {resumeOptions.length === 0 && !resumesLoading && (
              <p className="text-sm text-ink-secondary">No parsed resumes yet — upload one below.</p>
            )}

            <Button
              variant="outline"
              className="w-full"
              disabled={isBusy}
              isLoading={uploadAndApply.isPending}
              onClick={() => inputRef.current?.click()}
            >
              <Upload className="size-3.5" aria-hidden />
              Upload a new resume & apply
            </Button>

            <Button variant="ghost" className="w-full" disabled={isBusy} onClick={() => setMode("choice")}>
              Back
            </Button>
          </div>
        )}
      </Dialog>
    </>
  );
}
