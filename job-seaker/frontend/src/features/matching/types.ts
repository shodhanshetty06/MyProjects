/**
 * Matching feature — type contracts.
 *
 * Mirrors backend/app/modules/matching/schemas.py.
 */

export type WorkMode = "remote" | "hybrid" | "onsite" | "unknown";
export type EmploymentType =
  | "full_time"
  | "part_time"
  | "contract"
  | "internship"
  | "temporary"
  | "freelance"
  | "unknown";

export type MatchStatus = "pending" | "computed" | "stale" | "failed";
export type RecommendationType = "job" | "company" | "skill";
export type RecommendationStatus = "active" | "dismissed";

export interface MatchPreferenceInput {
  preferred_countries: string[];
  preferred_work_modes: WorkMode[];
  preferred_employment_types: EmploymentType[];
  years_of_experience: number | null;
  expected_salary_min: number | null;
  expected_salary_max: number | null;
  salary_currency: string | null;
  is_active: boolean;
}

export interface MatchPreference extends MatchPreferenceInput {
  id: string;
  user_id: string;
  created_at: string;
  updated_at: string;
}

export interface JobSummary {
  id: string;
  title: string;
  company_name: string;
  location: string | null;
  work_mode: WorkMode;
  employment_type: EmploymentType;
  salary_min: number | null;
  salary_max: number | null;
  salary_currency: string | null;
  source_url: string;
  posted_at: string | null;
}

export interface ScoreBreakdown {
  semantic_score: number;
  skill_match_score: number;
  rule_boost_score: number;
  boosts: Record<string, number | null>;
}

export interface ResumeJobMatch {
  id: string;
  resume_version_id: string;
  job_id: string;
  status: MatchStatus;
  match_score: number | null;
  matched_skills: string[];
  missing_skills: string[];
  strengths: string[];
  weaknesses: string[];
  score_breakdown: ScoreBreakdown;
  job: JobSummary | null;
  created_at: string;
  updated_at: string;
}

export interface Recommendation {
  id: string;
  recommendation_type: RecommendationType;
  job_id: string | null;
  company_id: string | null;
  skill_id: string | null;
  display_name: string;
  score: number;
  reason: string;
  status: RecommendationStatus;
  created_at: string;
}

export interface RefreshRecommendationsResponse {
  jobs: number;
  companies: number;
  skills: number;
}

export interface MatchingRealtimeEvent {
  type: "matches_computed" | "recommendations_updated" | "match_computed";
  [key: string]: unknown;
}
