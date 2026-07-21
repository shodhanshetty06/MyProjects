/**
 * Auto Apply feature - type contracts, mirroring
 * backend/app/modules/auto_apply/schemas.py.
 */
export type AutoApplyTaskStatus =
  | "pending"
  | "queued"
  | "running"
  | "submitted"
  | "requires_review"
  | "retrying"
  | "failed"
  | "cancelled";

export type AutoApplyAttemptStatus = "running" | "success" | "failed";
export type FailureCategory = "transient" | "validation" | "captcha" | "duplicate" | "platform_error" | "unknown";
export type ScreenshotStage = "before_submit" | "after_submit" | "missing_fields" | "step" | "error";
export type LogLevel = "info" | "warning" | "error";

export interface ApplicantProfileRead {
  id: string;
  user_id: string;
  auto_apply_enabled: boolean;
  first_name: string | null;
  last_name: string | null;
  email: string | null;
  phone: string | null;
  address_line1: string | null;
  city: string | null;
  state: string | null;
  postal_code: string | null;
  country: string | null;
  linkedin_url: string | null;
  portfolio_url: string | null;
  github_url: string | null;
  website_url: string | null;
  work_authorized: boolean | null;
  requires_sponsorship: boolean | null;
  willing_to_relocate: boolean | null;
  years_of_experience: number | null;
  desired_salary: number | null;
  notice_period: string | null;
  gender: string | null;
  race_ethnicity: string | null;
  veteran_status: string | null;
  disability_status: string | null;
  default_ats_resume_id: string | null;
  default_resume_id: string | null;
  custom_answers: Record<string, string>;
  created_at: string;
  updated_at: string;
}

export type ApplicantProfileUpsertRequest = Omit<ApplicantProfileRead, "id" | "user_id" | "created_at" | "updated_at">;

export interface AutoApplyTaskCreateRequest {
  job_id: string;
  ats_resume_id?: string | null;
  resume_version_id?: string | null;
}

export interface AutoApplyTaskRead {
  id: string;
  user_id: string;
  job_id: string | null;
  company_name: string;
  job_title: string;
  job_url: string;
  platform: string;
  status: AutoApplyTaskStatus;
  attempt_count: number;
  max_attempts: number;
  resume_used_display_name: string | null;
  missing_fields: string[];
  confirmation_detected: boolean;
  confirmation_text: string | null;
  submitted_at: string | null;
  tracking_application_id: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface AutoApplyAttemptSummary {
  id: string;
  attempt_number: number;
  status: AutoApplyAttemptStatus;
  failure_category: FailureCategory | null;
  started_at: string;
  finished_at: string | null;
  error_message: string | null;
}

export interface AutoApplyTaskDetail extends AutoApplyTaskRead {
  field_mapping_snapshot: Record<string, unknown>;
  attempts: AutoApplyAttemptSummary[];
}

export interface AutoApplyScreenshotRead {
  id: string;
  stage: ScreenshotStage;
  created_at: string;
}

export interface AutoApplyLogEntryRead {
  id: string;
  level: LogLevel;
  message: string;
  context: Record<string, unknown>;
  created_at: string;
}

export interface AutoApplyAttemptDetail extends AutoApplyAttemptSummary {
  screenshots: AutoApplyScreenshotRead[];
  logs: AutoApplyLogEntryRead[];
}

export interface AutoApplyTaskFiltersParams {
  status?: AutoApplyTaskStatus[];
  job_id?: string;
  page?: number;
  page_size?: number;
}
