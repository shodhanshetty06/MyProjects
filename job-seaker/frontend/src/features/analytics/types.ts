/**
 * Analytics feature — type contracts.
 *
 * Mirrors backend/app/modules/analytics/schemas.py.
 */

export type TrendInterval = "day" | "week" | "month";

export type ApplicationStatus =
  | "applied"
  | "interview"
  | "assessment"
  | "rejected"
  | "offer"
  | "accepted"
  | "withdrawn";

export interface AnalyticsFiltersParams {
  date_from?: string;
  date_to?: string;
  interval?: TrendInterval;
  company_name?: string;
  top_n?: number;
}

export interface TrendPoint {
  bucket: string;
  count: number;
}

export interface TrendSeries {
  interval: TrendInterval;
  points: TrendPoint[];
}

export interface AcceptanceRatePoint {
  bucket: string;
  offers: number;
  accepted: number;
  acceptance_rate: number;
}

export interface AcceptanceRateSeries {
  interval: TrendInterval;
  points: AcceptanceRatePoint[];
  overall_offers: number;
  overall_accepted: number;
  overall_acceptance_rate: number;
}

export interface ResumePerformance {
  total_resumes: number;
  total_versions: number;
  parsed_versions: number;
  parse_success_rate: number;
  average_skills_per_version: number;
  versions_trend: TrendPoint[];
}

export interface ScoreDistributionBucket {
  range_label: string;
  count: number;
}

export interface ATSScoreTrendPoint {
  bucket: string;
  average_ats_score: number | null;
  average_keyword_score: number | null;
  sample_size: number;
}

export interface ATSResumePerformance {
  total_ats_resumes: number;
  total_versions: number;
  completed_versions: number;
  completion_rate: number;
  average_ats_score: number | null;
  average_keyword_score: number | null;
  score_trend: ATSScoreTrendPoint[];
  score_distribution: ScoreDistributionBucket[];
}

export interface SkillCount {
  skill: string;
  count: number;
}

export interface CompanyOutcome {
  company_name: string;
  applications: number;
  interviews: number;
  offers: number;
}

export interface CountryCount {
  country: string;
  count: number;
}

export interface HiringTrendPoint {
  bucket: string;
  jobs_posted: number;
}

export interface HiringCompanyCount {
  company_name: string;
  jobs_posted: number;
}

export interface HiringTrends {
  interval: TrendInterval;
  trend: HiringTrendPoint[];
  top_hiring_companies: HiringCompanyCount[];
}

export interface SkillDemandItem {
  skill: string;
  job_count: number;
  user_has_skill: boolean;
}

export interface ResponseTimePoint {
  bucket: string;
  average_days: number | null;
  sample_size: number;
}

export interface ResponseTimeSeries {
  interval: TrendInterval;
  overall_average_days: number | null;
  points: ResponseTimePoint[];
}

export interface HeatmapCell {
  date: string;
  count: number;
}

export interface ApplicationHeatmap {
  cells: HeatmapCell[];
  total: number;
  current_streak_days: number;
  longest_streak_days: number;
}

export interface AnalyticsSummary {
  total_applications: number;
  interview_rate: number;
  offer_rate: number;
  acceptance_rate: number;
  average_response_days: number | null;
}

export interface AnalyticsDashboard {
  generated_at: string;
  date_from: string | null;
  date_to: string | null;
  interval: TrendInterval;
  summary: AnalyticsSummary;
  applications_trend: TrendSeries;
  interviews_trend: TrendSeries;
  offers_trend: TrendSeries;
  acceptance_rate: AcceptanceRateSeries;
  resume_performance: ResumePerformance;
  ats_resume_performance: ATSResumePerformance;
  top_skills: SkillCount[];
  top_companies: CompanyOutcome[];
  top_countries: CountryCount[];
  hiring_trends: HiringTrends;
  skill_demand: SkillDemandItem[];
  response_time: ResponseTimeSeries;
  heatmap: ApplicationHeatmap;
}
