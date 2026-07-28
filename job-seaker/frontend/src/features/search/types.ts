/**
 * Search feature - type contracts, mirroring
 * backend/app/modules/search/schemas.py. The original 5 search endpoints
 * are public (no auth header required); the job-feed/saved-jobs endpoints
 * added below are authenticated (JWT bearer via the shared `apiClient`,
 * same as every other feature — see api.ts's docstring).
 */
import type { NextBestAction } from "@/features/workspace/types";
export type SearchMode = "keyword" | "semantic" | "hybrid";
export type SearchSortBy = "relevance" | "salary" | "match_score" | "newest" | "company" | "ranked";
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

export interface JobQualityReason {
  label: string;
  positive: boolean;
  weight: number;
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
  /** 0-100, rule-based (backend `job_ingestion/quality.py`); null until the
   * ingestion pipeline's enrichment stage has scored this job. */
  quality_score: number | null;
  quality_reasons: JobQualityReason[];
  seniority: string;
  tech_stack: string[];
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

/**
 * Structured filters extracted from one natural-language query by
 * `GET /search/parse-query` (see backend `search/nl_query.py`). Every
 * filter field is `null`/empty when the query gave no signal for it.
 */
export interface ParsedSearchQuery {
  work_mode: WorkMode | null;
  skills: string[];
  salary_min: number | null;
  salary_max: number | null;
  salary_currency: string | null;
  salary_period: string | null;
  experience_level: ExperienceLevel | null;
  employment_type: EmploymentType | null;
  country: string | null;
  city: string | null;
  cleaned_query: string;
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

// --- Job Feed (authenticated) ---------------------------------------------

/**
 * Mirrors `search/schemas.py::JobFeedMode`. `recommended` is the default
 * landing tab; the backend falls back to `newest` server-side if the user
 * has zero computed matches yet, so the frontend never has to special-case
 * an empty "no matches" state for this mode.
 */
export type JobFeedMode = "recommended" | "all" | "newest" | "trending" | "saved" | "applied";

export const JOB_FEED_MODES: { value: JobFeedMode; label: string }[] = [
  { value: "recommended", label: "Recommended For You" },
  { value: "all", label: "All Jobs" },
  { value: "newest", label: "Newest" },
  { value: "trending", label: "Trending" },
  { value: "saved", label: "Saved Jobs" },
  { value: "applied", label: "Applied Jobs" },
];

export type ApplicationDifficulty = "easy" | "moderate" | "hard";

/** Mirrors `matching.ResumeJobMatch` fields exposed on a job feed item. */
export interface JobMatchContext {
  match_score: number | null;
  matched_skills: string[];
  missing_skills: string[];
  strengths: string[];
  weaknesses: string[];
  score_breakdown: Record<string, unknown>;
}

/**
 * Heuristic v1 signals (search/heuristics.py) — swappable placeholders for
 * hiring probability and application difficulty until a real model exists.
 */
export interface JobAISignals {
  hiring_probability: number;
  application_difficulty: ApplicationDifficulty;
  application_difficulty_score: number;
  expected_ats_score: number | null;
}

/**
 * The single unified job object `JobCard`/`JobDetailPage` consume — base
 * `SearchResultItem` fields plus `description`/`source_url`, match context,
 * AI signals, and per-user save/application state.
 */
/** Per-factor 0-100 inputs + final weighted score behind a job's rank on
 * the "Recommended" feed (see backend `search/ranking.py`). `null` on
 * every other feed mode. */
export interface RankingBreakdown {
  resume_match: number | null;
  hiring_probability: number;
  salary: number;
  recency: number;
  company_quality: number;
  competition: number;
  score: number;
}

export interface JobFeedItem extends SearchResultItem {
  description: string;
  source_url: string;
  match: JobMatchContext | null;
  ai_signals: JobAISignals;
  is_saved: boolean;
  application_status: string | null;
  ranking_breakdown: RankingBreakdown | null;
}

export interface SkillCheckItem {
  skill: string;
  matched: boolean;
}

export interface WhyThisScorePanel {
  matched_skills: string[];
  missing_skills: string[];
  skill_checklist: SkillCheckItem[];
  experience_match: string;
  education_match: string;
  location_match: string;
  salary_match: string;
  resume_suggestions: string[];
  narrative: string | null;
}

/**
 * `recommended_action` reuses `workspace.NextBestAction` shape by import
 * (mirrors the backend reusing `workspace.schemas.NextBestAction` rather
 * than redefining it).
 */
export interface JobSearchHomeSummary {
  greeting: string;
  jobs_matching_profile_count: number;
  new_jobs_since_yesterday: number;
  average_match_pct: number;
  highest_match_pct: number;
  companies_hiring_now: number;
  recommended_action: NextBestAction | null;
  strong_match_count: number;
  top_missing_skill: string | null;
  top_missing_skill_count: number;
  easy_apply_count: number;
  applications_needing_follow_up_count: number;
}

export interface SavedJob {
  id: string;
  user_id: string;
  job_id: string;
  created_at: string;
}

export interface JobFeedFiltersParams extends SearchFiltersParams {
  feed_mode?: JobFeedMode;
}

/**
 * `search:updates:{user_id}` channel, forwarded over the existing
 * `/workspace/ws` socket alongside workspace/matching events — see
 * `useSearchRealtime` in hooks.ts.
 */
export interface SearchRealtimeEvent {
  type: "new_matching_jobs" | "feed_stale" | "job_saved" | "job_unsaved";
  [key: string]: unknown;
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
