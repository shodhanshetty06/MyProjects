import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { Card } from "@/components/ui/Card";
import { Dropzone } from "@/components/ui/Dropzone";
import { useAuth } from "@/features/auth";
import { uploadResume } from "@/features/resume-parser/api";
import { paths } from "@/routes/paths";

/**
 * First screen a new account lands on, right after register auto-logs them
 * in. Asks exactly one thing — a resume — instead of a profile-detail form.
 * Uploading starts the auto-orchestrated parsing pipeline (existing
 * `uploadResume` -> `parse_resume_task` chain) and hands off to the Profile
 * Review screen, which waits for parsing to finish and shows the
 * AI-extracted data for the user to accept/edit/delete/add before
 * `POST /onboarding/complete` runs.
 */
export function WelcomePage() {
  const { user } = useAuth();
  const userId = user!.id;
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);

  const uploadMutation = useMutation({
    mutationFn: (file: File) =>
      uploadResume(userId, file, file.name.replace(/\.(pdf|docx)$/i, "")),
  });

  async function handleResumeUpload(file: File) {
    setError(null);
    try {
      const resume = await uploadMutation.mutateAsync(file);
      if (!resume.current_version) {
        throw new Error("Upload did not return a resume version.");
      }
      navigate(paths.onboardingReview(resume.id, resume.current_version.id), { replace: true });
    } catch {
      setError("Couldn't upload your resume — try again.");
    }
  }

  return (
    <div className="mx-auto w-full max-w-xl">
      <Card className="mt-4 p-6">
        <div className="space-y-4 text-center">
          <h1 className="text-lg font-semibold text-ink">Welcome! Let's get you set up.</h1>
          <p className="text-sm text-ink-secondary">Upload your resume to get started.</p>
          <Dropzone
            onFile={handleResumeUpload}
            disabled={uploadMutation.isPending}
            label={uploadMutation.isPending ? "Uploading..." : "Drop your resume here, or click to browse"}
          />
          {error && <p className="text-sm text-danger">{error}</p>}
        </div>
      </Card>
    </div>
  );
}
