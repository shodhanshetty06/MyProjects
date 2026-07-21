/**
 * Search feature - type contracts, mirroring
 * backend/app/modules/search/schemas.py. This module's endpoints are all
 * public (no auth header required).
 */
export type SearchMode = "keyword" | "semantic" | "hybrid";
export type SearchSortBy = "relevance" | "salary" | "match_score" | "newest" | "company";
export type SortDirection = "asc" | "desc";
export type ExperienceLevel = "internship" | "entry" | "mid" | "senior" | "lead" | "executive";
export type DatePosted = "any" | "last_24h" | "last_7d" | "last_30d";
export type WorkMode = "remote" | "hybrid" | "onsite" | "unknown";
export type EmploymentType =
  | "full_time"
  | "part_time"
  | "contract"
  | "internship"
  | "temporary"
  | "freelance"
  | "unknown";

export interface SkillTag {
  id: string;
  name: string;
}

export interface SearchResultCompany {
  id: string;
  name: string;
  logo_url: string | null;
  industry: string | null;
}

export interface SearchResultLocation {
  city: string | null;
  region: string | null;
  country: string | null;
}

export interface SearchResultItem {
  id: string;
  title: string;
  company: SearchResultCompany;
  location: SearchResultLocation | null;
  work_mode: WorkMode;
  employment_type: EmploymentType;
  salary_min: number | null;
  salary_max: number | null;
  salary_currency: string | null;
  salary_period: string | null;
  posted_at: string | null;
  skills: SkillTag[];
  experience_level: ExperienceLevel | null;
  relevance_score: number | null;
  match_score: number | null;
  keyword_score: number | null;
}

export interface AutocompleteSuggestion {
  type: "job_title" | "company" | "skill";
  label: string;
  id: string | null;
}

export interface SearchSuggestion {
  label: string;
  type: "skill" | "job_title";
}

export interface TrendingSkill {
  skill_id: string;
  name: string;
  job_count: number;
}

export interface TrendingJob {
  id: string;
  title: string;
  company_name: string;
  posted_at: string | null;
}

export interface SearchFiltersParams {
  q?: string;
  mode?: SearchMode;
  country?: string[];
  state?: string[];
  city?: string[];
  work_mode?: WorkMode[];
  salary_min?: number;
  salary_max?: number;
  salary_currency?: string;
  experience_level?: ExperienceLevel[];
  company_id?: string[];
  company_name?: string;
  skills?: string[];
  skills_match_all?: boolean;
  employment_type?: EmploymentType[];
  date_posted?: DatePosted;
  sort_by?: SearchSortBy;
  sort_direction?: SortDirection;
  page?: number;
  page_size?: number;
}

export const EXPERIENCE_LEVELS: ExperienceLevel[] = ["internship", "entry", "mid", "senior", "lead", "executive"];
export const WORK_MODES: WorkMode[] = ["remote", "hybrid", "onsite"];
export const EMPLOYMENT_TYPES: EmploymentType[] = [
  "full_time",
  "part_time",
  "contract",
  "internship",
  "temporary",
  "freelance",
];
export const DATE_POSTED_OPTIONS: DatePosted[] = ["any", "last_24h", "last_7d", "last_30d"];
