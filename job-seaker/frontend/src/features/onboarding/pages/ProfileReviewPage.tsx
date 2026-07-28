import { useMutation, useQuery } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import { Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import { Textarea } from "@/components/ui/Textarea";
import { useAuth } from "@/features/auth";
import { completeOnboarding } from "@/features/onboarding/api";
import {
  CertificationsSection,
  EducationSection,
  ExperienceSection,
  LanguagesSection,
  ProjectsSection,
} from "@/features/onboarding/components/ProfileReviewSections";
import { ProcessingStep } from "@/features/onboarding/components/ProcessingStep";
import { ReviewBadge } from "@/features/onboarding/components/ReviewBadge";
import { TagField } from "@/features/onboarding/components/TagField";
import {
  buildInitialReviewState,
  toParsedResumeData,
  type ProfileReviewState,
} from "@/features/onboarding/reviewState";
import { getResumeVersion, updateResumeParsedData } from "@/features/resume-parser/api";
import { useParsingStageMessage } from "@/features/resume-parser/hooks";
import type { ResumeVersionStatus } from "@/features/resume-parser/types";
import { paths } from "@/routes/paths";

const POLL_STATUSES: ResumeVersionStatus[] = ["uploaded", "parsing"];

/**
 * The "AI-first" heart of onboarding: instead of a manual profile form, the
 * user reviews what the resume parser already extracted. Every field is
 * editable; list sections (experience, education, ...) additionally carry
 * Accept/Delete per entry and an Add button for the whole section, so the
 * user can accept, correct, remove, or add to what the AI produced.
 * "Complete Profile" persists the edits back onto the parsed-data row and
 * then completes onboarding.
 */
export function ProfileReviewPage() {
  const { user } = useAuth();
  const userId = user!.id;
  const { resumeId, versionId } = useParams<{ resumeId: string; versionId: string }>();
  const navigate = useNavigate();

  const versionQuery = useQuery({
    queryKey: ["resume-version", userId, resumeId, versionId],
    queryFn: () => getResumeVersion(userId, resumeId as string, versionId as string),
    enabled: !!resumeId && !!versionId,
    refetchInterval: (query) => (query.state.data && POLL_STATUSES.includes(query.state.data.status) ? 2500 : false),
  });
  const version = versionQuery.data;
  const stageMessage = useParsingStageMessage(version?.status);

  const [review, setReview] = useState<ProfileReviewState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitted, setSubmitted] = useState(false);
  const [finished, setFinished] = useState(false);

  useEffect(() => {
    if (!review && version?.status === "parsed" && version.parsed_data) {
      setReview(buildInitialReviewState(version.parsed_data));
    }
  }, [review, version]);

  const completeMutation = useMutation({
    mutationFn: async () => {
      if (!review) return;
      await updateResumeParsedData(userId, resumeId as string, versionId as string, toParsedResumeData(review));
      await completeOnboarding({
        target_roles: review.targetRoles,
        work_authorization: review.workAuthorization.trim() || null,
        preferred_countries: [],
        preferred_work_modes: [],
        preferred_employment_types: [],
        years_of_experience: null,
        expected_salary_min: null,
        expected_salary_max: null,
        salary_currency: null,
        is_active: true,
      });
    },
    onSuccess: () => setSubmitted(true),
    onError: () => setError("Couldn't save your profile — try again."),
  });

  function update<K extends keyof ProfileReviewState>(key: K, value: ProfileReviewState[K]) {
    setReview((r) => (r ? { ...r, [key]: value } : r));
  }

  if (submitted) {
    return (
      <div className="mx-auto w-full max-w-xl">
        <Card className="mt-4 p-6">
          <ProcessingStep
            userId={userId}
            onComplete={() => {
              if (!finished) {
                setFinished(true);
                navigate(paths.workspace, { replace: true });
              }
            }}
          />
        </Card>
      </div>
    );
  }

  if (version?.status === "failed") {
    return (
      <div className="mx-auto w-full max-w-xl">
        <Card className="mt-4 space-y-3 p-6 text-center">
          <h1 className="text-lg font-semibold text-ink">We couldn't read that resume</h1>
          <p className="text-sm text-ink-secondary">
            {version.error_message ?? "Something went wrong parsing your file."}
          </p>
          <Button onClick={() => navigate(paths.onboarding, { replace: true })}>Try a different file</Button>
        </Card>
      </div>
    );
  }

  if (versionQuery.isError || !review) {
    return (
      <div className="mx-auto w-full max-w-xl">
        <Card className="mt-4 space-y-3 p-6 text-center">
          {versionQuery.isError ? (
            <p className="text-sm text-danger">Couldn't load your resume — try refreshing the page.</p>
          ) : (
            <>
              <Loader2 className="mx-auto size-6 animate-spin text-brand" aria-hidden />
              <h1 className="text-lg font-semibold text-ink">Reading your resume...</h1>
              <div className="h-5 overflow-hidden">
                <AnimatePresence mode="wait">
                  <motion.p
                    key={stageMessage}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -6 }}
                    transition={{ duration: 0.25 }}
                    className="text-sm text-ink-secondary"
                  >
                    {stageMessage}
                  </motion.p>
                </AnimatePresence>
              </div>
            </>
          )}
          <Button
            variant="ghost"
            onClick={() => navigate(paths.onboarding, { replace: true })}
          >
            Taking too long? Cancel and upload a different file
          </Button>
        </Card>
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-2xl space-y-6 pb-10">
      <div>
        <h1 className="text-lg font-semibold text-ink">Review your profile</h1>
        <p className="mt-1 text-sm text-ink-secondary">
          Here's what we pulled from your resume. Accept what looks right, edit anything that's off,
          delete what doesn't belong, and add whatever's missing.
        </p>
      </div>

      <Card className="space-y-3 p-5">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-ink">Summary</h2>
          <ReviewBadge state={review.summary.trim() ? "ai" : "missing"} />
        </div>
        <Textarea value={review.summary} onChange={(e) => update("summary", e.target.value)} />
      </Card>

      <Card className="space-y-3 p-5">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-ink">Target roles</h2>
          <ReviewBadge state={review.targetRoles.length ? "ai" : "missing"} />
        </div>
        <p className="text-xs text-ink-muted">Suggested from your work history — add or remove as you like.</p>
        <TagField
          values={review.targetRoles}
          onChange={(targetRoles) => update("targetRoles", targetRoles)}
          placeholder="e.g. Backend Engineer"
        />
      </Card>

      <Card className="space-y-3 p-5">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-ink">Work authorization</h2>
          <ReviewBadge state={review.workAuthorization.trim() ? "added" : "missing"} />
        </div>
        <p className="text-xs text-ink-muted">Your resume didn't mention this — add it so matches respect it.</p>
        <Input
          value={review.workAuthorization}
          onChange={(e) => update("workAuthorization", e.target.value)}
          placeholder="e.g. US Citizen, H1-B, Requires sponsorship"
        />
      </Card>

      <Card className="space-y-3 p-5">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-ink">Skills</h2>
          <ReviewBadge state={review.skills.length ? "ai" : "missing"} />
        </div>
        <TagField values={review.skills} onChange={(skills) => update("skills", skills)} placeholder="Add a skill" />
      </Card>

      <Card className="p-5">
        <ExperienceSection items={review.experience} onChange={(experience) => update("experience", experience)} />
      </Card>

      <Card className="p-5">
        <EducationSection items={review.education} onChange={(education) => update("education", education)} />
      </Card>

      <Card className="p-5">
        <ProjectsSection items={review.projects} onChange={(projects) => update("projects", projects)} />
      </Card>

      <Card className="p-5">
        <CertificationsSection
          items={review.certifications}
          onChange={(certifications) => update("certifications", certifications)}
        />
      </Card>

      <Card className="space-y-3 p-5">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-ink">Achievements</h2>
          <ReviewBadge state={review.achievements.length ? "ai" : "missing"} />
        </div>
        <TagField
          values={review.achievements}
          onChange={(achievements) => update("achievements", achievements)}
          placeholder="Add an achievement"
        />
      </Card>

      <Card className="p-5">
        <LanguagesSection items={review.languages} onChange={(languages) => update("languages", languages)} />
      </Card>

      {error && <p className="text-sm text-danger">{error}</p>}

      <div className="flex justify-end gap-3">
        <Button
          variant="ghost"
          size="lg"
          onClick={() => navigate(paths.onboarding, { replace: true })}
          disabled={completeMutation.isPending}
        >
          Cancel
        </Button>
        <Button onClick={() => completeMutation.mutate()} isLoading={completeMutation.isPending} size="lg">
          Complete Profile
        </Button>
      </div>
    </div>
  );
}
