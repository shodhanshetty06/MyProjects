/**
 * Onboarding feature — type contracts, mirroring
 * backend/app/modules/user_profile/schemas.py.
 */
import type { EmploymentType, WorkMode } from "@/features/matching/types";

export interface OnboardingCompletePayload {
  target_roles: string[];
  work_authorization: string | null;
  preferred_countries: string[];
  preferred_work_modes: WorkMode[];
  preferred_employment_types: EmploymentType[];
  years_of_experience: number | null;
  expected_salary_min: number | null;
  expected_salary_max: number | null;
  salary_currency: string | null;
  is_active: boolean;
}

export interface UserProfileRead {
  id: string;
  user_id: string;
  target_roles: string[];
  work_authorization: string | null;
  onboarding_completed_at: string | null;
}

export interface OnboardingStatusRead {
  onboarding_completed: boolean;
  has_resume: boolean;
  pipeline_run_id: string | null;
  pipeline_status: string | null;
}

export interface OnboardingCompleteResponse {
  profile: UserProfileRead;
  pipeline_run_id: string | null;
}
