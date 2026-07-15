/**
 * AI Career Coach feature — type contracts.
 *
 * Mirrors backend/app/modules/ai_career_coach/schemas.py.
 */

export type Priority = "high" | "medium" | "low";

export type CareerInsightType =
  | "resume_review"
  | "career_suggestions"
  | "skill_gap_analysis"
  | "learning_recommendations"
  | "salary_prediction"
  | "career_roadmap"
  | "interview_preparation"
  | "job_readiness_score"
  | "market_trend_analysis"
  | "country_demand";

export interface SectionFeedback {
  section: string;
  feedback: string;
}

export interface ImprovementSuggestion {
  section: string;
  issue: string;
  suggestion: string;
  priority: Priority;
}

export interface ResumeReviewResult {
  overall_score: number;
  summary: string;
  strengths: string[];
  weaknesses: string[];
  section_feedback: SectionFeedback[];
  improvement_suggestions: ImprovementSuggestion[];
}

export interface SuggestedRole {
  title: string;
  fit_score: number;
  match_reason: string;
}

export interface CareerSuggestionResult {
  summary: string;
  suggested_roles: SuggestedRole[];
}

export interface SkillGapItem {
  skill: string;
  importance: Priority;
  reason: string;
}

export interface SkillGapResult {
  target_role: string;
  matched_skills: string[];
  missing_skills: SkillGapItem[];
  overall_readiness_percent: number;
}

export interface LearningRecommendation {
  skill: string;
  resource_type: "course" | "certification" | "book" | "project" | "practice";
  title: string;
  provider: string | null;
  description: string;
  priority: Priority;
}

export interface LearningRecommendationResult {
  target_role: string;
  recommendations: LearningRecommendation[];
}

export interface SalaryPredictionResult {
  target_role: string;
  country: string | null;
  predicted_min: number;
  predicted_max: number;
  currency: string;
  based_on_market_data: boolean;
  sample_size: number;
  narrative: string;
}

export interface RoadmapMilestone {
  phase: string;
  goals: string[];
  skills_to_develop: string[];
  suggested_actions: string[];
}

export interface CareerRoadmapResult {
  target_role: string;
  timeframe_months: number;
  milestones: RoadmapMilestone[];
}

export interface InterviewQuestion {
  question: string;
  category: "behavioral" | "technical" | "situational";
  tip: string;
}

export interface InterviewPrepResult {
  target_role: string;
  questions: InterviewQuestion[];
  general_tips: string[];
}

export interface JobReadinessScoreResult {
  overall_score: number;
  resume_quality_score: number;
  skill_match_score: number;
  completeness_score: number;
  narrative: string;
  recommendations: string[];
}

export interface SkillDemand {
  skill: string;
  job_count: number;
}

export interface CompanyDemand {
  company: string;
  job_count: number;
}

export interface MarketTrendResult {
  window_days: number;
  title: string | null;
  total_postings: number;
  previous_window_postings: number;
  growth_percent: number | null;
  top_skills: SkillDemand[];
  top_companies: CompanyDemand[];
  narrative: string;
}

export interface CountryDemand {
  country: string;
  country_code: string | null;
  job_count: number;
  avg_salary_min: number | null;
  avg_salary_max: number | null;
  currency: string | null;
}

export interface CountryDemandResult {
  title: string | null;
  skill: string | null;
  countries: CountryDemand[];
  narrative: string;
}

export interface CareerInsightSummary {
  id: string;
  insight_type: CareerInsightType;
  resume_version_id: string | null;
  created_at: string;
}

export interface CareerInsightRead extends CareerInsightSummary {
  input_params: Record<string, unknown>;
  result: Record<string, unknown>;
  model_used: string;
}

export type CoachMessageRole = "user" | "assistant";

export interface CoachMessageRead {
  id: string;
  role: CoachMessageRole;
  content: string;
  created_at: string;
}

export interface CoachConversationRead {
  id: string;
  title: string;
  last_message_at: string | null;
  created_at: string;
}

export interface CoachConversationDetail extends CoachConversationRead {
  messages: CoachMessageRead[];
}

export interface ChatResponse {
  conversation: CoachConversationRead;
  message: CoachMessageRead;
}
