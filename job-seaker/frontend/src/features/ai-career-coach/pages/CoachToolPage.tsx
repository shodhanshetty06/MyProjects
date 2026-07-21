import { useMutation } from "@tanstack/react-query";
import {
  Award,
  Compass,
  DollarSign,
  FileSearch,
  Globe2,
  GraduationCap,
  ListChecks,
  MessageCircleQuestion,
  Sparkles,
  TrendingUp,
  type LucideIcon,
} from "lucide-react";
import { useState } from "react";
import { useParams } from "react-router-dom";

import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/Button";
import { Card, CardContent } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import {
  analyzeCountryDemand,
  analyzeMarketTrends,
  analyzeSkillGap,
  computeJobReadiness,
  generateRoadmap,
  predictSalary,
  prepareInterview,
  recommendLearning,
  reviewResume,
  suggestCareers,
} from "@/features/ai-career-coach/api";
import { ThinkingIndicator } from "@/features/ai-career-coach/components/ThinkingIndicator";
import {
  CareerSuggestionsView,
  CountryDemandView,
  InterviewPrepView,
  JobReadinessView,
  LearningRecommendationsView,
  MarketTrendsView,
  ResumeReviewView,
  RoadmapView,
  SalaryPredictionView,
  SkillGapView,
} from "@/features/ai-career-coach/components/ToolResults";
import { useAuth } from "@/features/auth";
import { useParsedResumeOptions } from "@/features/resume-parser/hooks";

interface ToolMeta {
  label: string;
  icon: LucideIcon;
  description: string;
  needsResume: boolean;
  needsRole: boolean;
}

const TOOL_META: Record<string, ToolMeta> = {
  "skill-gap": { label: "Skill Gap Analysis", icon: ListChecks, description: "Compare your resume against a target role.", needsResume: true, needsRole: true },
  roadmap: { label: "Career Roadmap", icon: Compass, description: "A milestone-by-milestone plan toward a target role.", needsResume: true, needsRole: true },
  "salary-prediction": { label: "Salary Prediction", icon: DollarSign, description: "Estimated compensation range grounded in market data.", needsResume: false, needsRole: true },
  "learning-recommendations": { label: "Learning Recommendations", icon: GraduationCap, description: "Courses, projects, and certifications to close your gaps.", needsResume: true, needsRole: true },
  "interview-prep": { label: "Interview Prep", icon: MessageCircleQuestion, description: "Likely questions and tips for your target role.", needsResume: true, needsRole: true },
  "job-readiness": { label: "Job Readiness Score", icon: Award, description: "How ready your resume is to apply, right now.", needsResume: true, needsRole: false },
  "resume-review": { label: "Resume Review", icon: FileSearch, description: "A structured critique of your current resume.", needsResume: true, needsRole: false },
  "career-suggestions": { label: "Career Suggestions", icon: Sparkles, description: "Roles that fit your experience, ranked by fit.", needsResume: true, needsRole: false },
  "market-trends": { label: "Market Trends", icon: TrendingUp, description: "Posting volume and top skills for a role.", needsResume: false, needsRole: false },
  "country-demand": { label: "Country Demand", icon: Globe2, description: "Where a role or skill is in demand globally.", needsResume: false, needsRole: false },
};

export function CoachToolPage() {
  const { toolId } = useParams<{ toolId: string }>();
  const { user } = useAuth();
  const userId = user!.id;
  const meta = toolId ? TOOL_META[toolId] : undefined;

  const [resumeVersionId, setResumeVersionId] = useState("");
  const [targetRole, setTargetRole] = useState("");
  const [timeframeMonths, setTimeframeMonths] = useState(6);
  const [country, setCountry] = useState("");
  const [yearsExperience, setYearsExperience] = useState("");
  const [title, setTitle] = useState("");
  const [skill, setSkill] = useState("");

  const { options: resumeOptions, isLoading: resumesLoading } = useParsedResumeOptions(userId);

  const mutation = useMutation({
    mutationFn: async () => {
      switch (toolId) {
        case "skill-gap":
          return { kind: "skill-gap", data: await analyzeSkillGap(userId, resumeVersionId, targetRole) } as const;
        case "roadmap":
          return { kind: "roadmap", data: await generateRoadmap(userId, resumeVersionId, targetRole, timeframeMonths) } as const;
        case "salary-prediction":
          return {
            kind: "salary-prediction",
            data: await predictSalary(userId, targetRole, country || null, yearsExperience ? Number(yearsExperience) : null, resumeVersionId || null),
          } as const;
        case "learning-recommendations":
          return { kind: "learning-recommendations", data: await recommendLearning(userId, resumeVersionId, targetRole) } as const;
        case "interview-prep":
          return { kind: "interview-prep", data: await prepareInterview(userId, resumeVersionId, targetRole) } as const;
        case "job-readiness":
          return { kind: "job-readiness", data: await computeJobReadiness(userId, resumeVersionId, targetRole || null) } as const;
        case "resume-review":
          return { kind: "resume-review", data: await reviewResume(userId, resumeVersionId) } as const;
        case "career-suggestions":
          return { kind: "career-suggestions", data: await suggestCareers(userId, resumeVersionId) } as const;
        case "market-trends":
          return { kind: "market-trends", data: await analyzeMarketTrends(userId, title || null, 90) } as const;
        case "country-demand":
          return { kind: "country-demand", data: await analyzeCountryDemand(userId, title || null, skill || null) } as const;
        default:
          throw new Error("Unknown tool");
      }
    },
  });

  if (!meta) {
    return <EmptyState title="Unknown tool" />;
  }

  const canRun =
    (!meta.needsResume || !!resumeVersionId) &&
    (!meta.needsRole || targetRole.trim().length > 0) &&
    (toolId !== "salary-prediction" || targetRole.trim().length > 0);

  return (
    <div className="mx-auto max-w-2xl">
      <PageHeader eyebrow="AI Career Coach" title={meta.label} description={meta.description} />

      <Card className="mb-6">
        <CardContent className="space-y-3 p-5">
          {meta.needsResume && (
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-ink-secondary">Resume</label>
              {resumesLoading ? (
                <p className="text-xs text-ink-muted">Loading resumes...</p>
              ) : resumeOptions.length === 0 ? (
                <p className="text-xs text-ink-muted">Upload and parse a resume first.</p>
              ) : (
                <Select value={resumeVersionId} onValueChange={setResumeVersionId} options={resumeOptions} placeholder="Choose a parsed resume" />
              )}
            </div>
          )}
          {toolId === "salary-prediction" && !meta.needsResume && resumeOptions.length > 0 && (
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-ink-secondary">Resume (optional)</label>
              <Select value={resumeVersionId} onValueChange={setResumeVersionId} options={resumeOptions} placeholder="None" />
            </div>
          )}
          {(meta.needsRole || toolId === "job-readiness") && (
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-ink-secondary">Target role{toolId === "job-readiness" ? " (optional)" : ""}</label>
              <Input value={targetRole} onChange={(e) => setTargetRole(e.target.value)} placeholder="e.g. Senior Frontend Engineer" />
            </div>
          )}
          {toolId === "roadmap" && (
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-ink-secondary">Timeframe (months)</label>
              <Input type="number" min={1} max={36} value={timeframeMonths} onChange={(e) => setTimeframeMonths(Number(e.target.value))} />
            </div>
          )}
          {toolId === "salary-prediction" && (
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-ink-secondary">Country (optional)</label>
                <Input value={country} onChange={(e) => setCountry(e.target.value)} placeholder="e.g. United States" />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-ink-secondary">Years experience (optional)</label>
                <Input type="number" min={0} value={yearsExperience} onChange={(e) => setYearsExperience(e.target.value)} />
              </div>
            </div>
          )}
          {(toolId === "market-trends" || toolId === "country-demand") && (
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-ink-secondary">Job title (optional)</label>
              <Input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g. Data Scientist" />
            </div>
          )}
          {toolId === "country-demand" && (
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-ink-secondary">Skill (optional)</label>
              <Input value={skill} onChange={(e) => setSkill(e.target.value)} placeholder="e.g. Kubernetes" />
            </div>
          )}

          <Button className="w-full" disabled={!canRun || mutation.isPending} onClick={() => mutation.mutate()}>
            <meta.icon className="size-4" aria-hidden />
            Run {meta.label}
          </Button>
        </CardContent>
      </Card>

      {mutation.isPending && (
        <Card>
          <CardContent className="p-6">
            <ThinkingIndicator label="Analyzing..." />
          </CardContent>
        </Card>
      )}

      {mutation.isError && <EmptyState title="Something went wrong" description="Try running this tool again." />}

      {mutation.data && (
        <div>
          {mutation.data.kind === "skill-gap" && <SkillGapView result={mutation.data.data} />}
          {mutation.data.kind === "roadmap" && <RoadmapView result={mutation.data.data} />}
          {mutation.data.kind === "salary-prediction" && <SalaryPredictionView result={mutation.data.data} />}
          {mutation.data.kind === "learning-recommendations" && <LearningRecommendationsView result={mutation.data.data} />}
          {mutation.data.kind === "interview-prep" && <InterviewPrepView result={mutation.data.data} />}
          {mutation.data.kind === "job-readiness" && <JobReadinessView result={mutation.data.data} />}
          {mutation.data.kind === "resume-review" && <ResumeReviewView result={mutation.data.data} />}
          {mutation.data.kind === "career-suggestions" && <CareerSuggestionsView result={mutation.data.data} />}
          {mutation.data.kind === "market-trends" && <MarketTrendsView result={mutation.data.data} />}
          {mutation.data.kind === "country-demand" && <CountryDemandView result={mutation.data.data} />}
        </div>
      )}
    </div>
  );
}

export { TOOL_META };
export type { ToolMeta };
