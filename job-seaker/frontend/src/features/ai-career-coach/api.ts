import { apiClient } from "@/api/client";
import type { BaseResponse, PaginatedResponse } from "@/types/common";

import type {
  CareerInsightRead,
  CareerInsightSummary,
  CareerInsightType,
  CareerRoadmapResult,
  CareerSuggestionResult,
  ChatResponse,
  CoachConversationDetail,
  CoachConversationRead,
  CountryDemandResult,
  InterviewPrepResult,
  JobReadinessScoreResult,
  LearningRecommendationResult,
  MarketTrendResult,
  ResumeReviewResult,
  SalaryPredictionResult,
  SkillGapResult,
} from "@/features/ai-career-coach/types";

/**
 * AI Career Coach feature — API bindings.
 *
 * `auth` has no real session yet (Phase 0 stub — see
 * backend/app/api/deps.py::get_current_user_id), so every call here takes an
 * explicit `userId` and sends it as the same trusted `X-User-Id` header the
 * backend expects, the same convention `matching`'s frontend already uses.
 */
function authHeaders(userId: string) {
  return { headers: { "X-User-Id": userId } };
}

export async function reviewResume(
  userId: string,
  resumeVersionId: string,
): Promise<ResumeReviewResult> {
  const response = await apiClient.post<BaseResponse<ResumeReviewResult>>(
    `/career-coach/resumes/${resumeVersionId}/review`,
    undefined,
    authHeaders(userId),
  );
  return response.data.data as ResumeReviewResult;
}

export async function suggestCareers(
  userId: string,
  resumeVersionId: string,
): Promise<CareerSuggestionResult> {
  const response = await apiClient.post<BaseResponse<CareerSuggestionResult>>(
    `/career-coach/resumes/${resumeVersionId}/career-suggestions`,
    undefined,
    authHeaders(userId),
  );
  return response.data.data as CareerSuggestionResult;
}

export async function analyzeSkillGap(
  userId: string,
  resumeVersionId: string,
  targetRole: string,
): Promise<SkillGapResult> {
  const response = await apiClient.post<BaseResponse<SkillGapResult>>(
    `/career-coach/resumes/${resumeVersionId}/skill-gap`,
    { target_role: targetRole },
    authHeaders(userId),
  );
  return response.data.data as SkillGapResult;
}

export async function recommendLearning(
  userId: string,
  resumeVersionId: string,
  targetRole: string,
): Promise<LearningRecommendationResult> {
  const response = await apiClient.post<BaseResponse<LearningRecommendationResult>>(
    `/career-coach/resumes/${resumeVersionId}/learning-recommendations`,
    { target_role: targetRole },
    authHeaders(userId),
  );
  return response.data.data as LearningRecommendationResult;
}

export async function generateRoadmap(
  userId: string,
  resumeVersionId: string,
  targetRole: string,
  timeframeMonths: number,
): Promise<CareerRoadmapResult> {
  const response = await apiClient.post<BaseResponse<CareerRoadmapResult>>(
    `/career-coach/resumes/${resumeVersionId}/roadmap`,
    { target_role: targetRole, timeframe_months: timeframeMonths },
    authHeaders(userId),
  );
  return response.data.data as CareerRoadmapResult;
}

export async function prepareInterview(
  userId: string,
  resumeVersionId: string,
  targetRole: string,
): Promise<InterviewPrepResult> {
  const response = await apiClient.post<BaseResponse<InterviewPrepResult>>(
    `/career-coach/resumes/${resumeVersionId}/interview-prep`,
    { target_role: targetRole },
    authHeaders(userId),
  );
  return response.data.data as InterviewPrepResult;
}

export async function computeJobReadiness(
  userId: string,
  resumeVersionId: string,
  targetRole: string | null,
): Promise<JobReadinessScoreResult> {
  const response = await apiClient.post<BaseResponse<JobReadinessScoreResult>>(
    `/career-coach/resumes/${resumeVersionId}/job-readiness`,
    { target_role: targetRole },
    authHeaders(userId),
  );
  return response.data.data as JobReadinessScoreResult;
}

export async function predictSalary(
  userId: string,
  targetRole: string,
  country: string | null,
  yearsExperience: number | null,
  resumeVersionId: string | null,
): Promise<SalaryPredictionResult> {
  const response = await apiClient.post<BaseResponse<SalaryPredictionResult>>(
    "/career-coach/salary-prediction",
    {
      target_role: targetRole,
      country,
      years_experience: yearsExperience,
      resume_version_id: resumeVersionId,
    },
    authHeaders(userId),
  );
  return response.data.data as SalaryPredictionResult;
}

export async function analyzeMarketTrends(
  userId: string,
  title: string | null,
  windowDays = 90,
): Promise<MarketTrendResult> {
  const response = await apiClient.get<BaseResponse<MarketTrendResult>>(
    "/career-coach/market-trends",
    { ...authHeaders(userId), params: { title: title || undefined, window_days: windowDays } },
  );
  return response.data.data as MarketTrendResult;
}

export async function analyzeCountryDemand(
  userId: string,
  title: string | null,
  skill: string | null,
): Promise<CountryDemandResult> {
  const response = await apiClient.get<BaseResponse<CountryDemandResult>>(
    "/career-coach/country-demand",
    { ...authHeaders(userId), params: { title: title || undefined, skill: skill || undefined } },
  );
  return response.data.data as CountryDemandResult;
}

export async function listInsights(
  userId: string,
  insightType?: CareerInsightType,
  page = 1,
  pageSize = 20,
): Promise<PaginatedResponse<CareerInsightSummary>> {
  const response = await apiClient.get<PaginatedResponse<CareerInsightSummary>>(
    "/career-coach/insights",
    { ...authHeaders(userId), params: { insight_type: insightType, page, page_size: pageSize } },
  );
  return response.data;
}

export async function getInsight(userId: string, insightId: string): Promise<CareerInsightRead> {
  const response = await apiClient.get<BaseResponse<CareerInsightRead>>(
    `/career-coach/insights/${insightId}`,
    authHeaders(userId),
  );
  return response.data.data as CareerInsightRead;
}

export async function sendChatMessage(
  userId: string,
  message: string,
  conversationId: string | null,
): Promise<ChatResponse> {
  const response = await apiClient.post<BaseResponse<ChatResponse>>(
    "/career-coach/chat",
    { message, conversation_id: conversationId },
    authHeaders(userId),
  );
  return response.data.data as ChatResponse;
}

export async function listConversations(
  userId: string,
  page = 1,
  pageSize = 20,
): Promise<PaginatedResponse<CoachConversationRead>> {
  const response = await apiClient.get<PaginatedResponse<CoachConversationRead>>(
    "/career-coach/conversations",
    { ...authHeaders(userId), params: { page, page_size: pageSize } },
  );
  return response.data;
}

export async function getConversation(
  userId: string,
  conversationId: string,
): Promise<CoachConversationDetail> {
  const response = await apiClient.get<BaseResponse<CoachConversationDetail>>(
    `/career-coach/conversations/${conversationId}`,
    authHeaders(userId),
  );
  return response.data.data as CoachConversationDetail;
}

export async function deleteConversation(userId: string, conversationId: string): Promise<void> {
  await apiClient.delete(`/career-coach/conversations/${conversationId}`, authHeaders(userId));
}
