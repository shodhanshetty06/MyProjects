/**
 * Workspace feature — type contracts, mirroring
 * backend/app/modules/workspace/schemas.py.
 */
import type { CareerInsightSummary } from "@/features/ai-career-coach/types";
import type { Recommendation } from "@/features/matching/types";
import type { NotificationRead } from "@/features/notification/types";
import type { ResumeVersionSummary } from "@/features/resume-parser/types";
import type { DashboardResponse } from "@/features/tracking/types";

export type PipelineRunStatus = "pending" | "running" | "completed" | "failed";

export interface PipelineStep {
  name: string;
  status: "pending" | "running" | "completed" | "failed";
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
}

export interface PipelineStatus {
  run_id: string;
  resume_version_id: string;
  status: PipelineRunStatus;
  current_step: string | null;
  progress_pct: number;
  steps: PipelineStep[];
  error_message: string | null;
}

export interface ResumeHealth {
  resume_id: string | null;
  resume_version_id: string | null;
  resume_score: number | null;
  resume_score_breakdown: Record<string, boolean | number> | null;
  ats_baseline_score: number | null;
  ats_baseline_missing_skills: string[] | null;
  health_computed_at: string | null;
  career_health_score: number | null;
}

export interface NextBestAction {
  key: string;
  title: string;
  description: string;
  cta_path: string;
  priority: number;
}

export interface WorkspaceSummary {
  onboarding_completed: boolean;
  resume_health: ResumeHealth;
  recommended_jobs: Recommendation[];
  resume_versions: ResumeVersionSummary[];
  applications: DashboardResponse;
  coach_highlights: CareerInsightSummary[];
  notifications: NotificationRead[];
  unread_notification_count: number;
  next_best_actions: NextBestAction[];
  pipeline_status: PipelineStatus | null;
}

export interface WorkspaceRealtimeEvent {
  type:
    | "pipeline_progress"
    | "workspace_ready"
    | "ats_resume_ready"
    | "application_status_changed"
    | "matches_computed"
    | "recommendations_updated"
    | "match_computed";
  [key: string]: unknown;
}
