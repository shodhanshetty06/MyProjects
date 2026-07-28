import { apiClient } from "@/api/client";
import type {
  AutocompleteSuggestion,
  JobFeedFiltersParams,
  JobFeedItem,
  JobSearchHomeSummary,
  ParsedSearchQuery,
  SavedJob,
  SearchFiltersParams,
  SearchResultItem,
  SearchSuggestion,
  TrendingJob,
  TrendingSkill,
  WhyThisScorePanel,
} from "@/features/search/types";
import type { BaseResponse, PaginatedResponse } from "@/types/common";

/**
 * Search feature — API bindings. The original 5 endpoints below are public
 * (no auth header). The job-feed/saved-jobs endpoints are authenticated —
 * plain `apiClient` calls, since the shared client already attaches the
 * JWT bearer token + refresh cookie. Do NOT copy `matching/api.ts`'s stale
 * `X-User-Id` header pattern here.
 */
export async function searchJobs(filters: SearchFiltersParams): Promise<PaginatedResponse<SearchResultItem>> {
  const response = await apiClient.get<PaginatedResponse<SearchResultItem>>("/search", { params: filters });
  return response.data;
}

export async function getAutocomplete(q: string, limit = 10): Promise<AutocompleteSuggestion[]> {
  const response = await apiClient.get<BaseResponse<AutocompleteSuggestion[]>>("/search/autocomplete", {
    params: { q, limit },
  });
  return response.data.data ?? [];
}

export async function parseSearchQuery(q: string): Promise<ParsedSearchQuery> {
  const response = await apiClient.get<BaseResponse<ParsedSearchQuery>>("/search/parse-query", {
    params: { q },
  });
  return response.data.data as ParsedSearchQuery;
}

export async function getSearchSuggestions(q: string, limit = 8): Promise<SearchSuggestion[]> {
  const response = await apiClient.get<BaseResponse<SearchSuggestion[]>>("/search/suggestions", {
    params: { q, limit },
  });
  return response.data.data ?? [];
}

export async function getTrendingSkills(windowDays = 30, limit = 20): Promise<TrendingSkill[]> {
  const response = await apiClient.get<BaseResponse<TrendingSkill[]>>("/search/trending/skills", {
    params: { window_days: windowDays, limit },
  });
  return response.data.data ?? [];
}

export async function getTrendingJobs(windowDays = 30, limit = 20): Promise<TrendingJob[]> {
  const response = await apiClient.get<BaseResponse<TrendingJob[]>>("/search/trending/jobs", {
    params: { window_days: windowDays, limit },
  });
  return response.data.data ?? [];
}

/**
 * Phase 0 module-connectivity stub, kept only because nothing has removed
 * its last caller yet.
 */
export async function getSearchModuleStatus(): Promise<BaseResponse> {
  const response = await apiClient.get<BaseResponse>("/search/status");
  return response.data;
}

// --- Job Feed (authenticated) ---------------------------------------------

export async function getJobFeed(filters: JobFeedFiltersParams): Promise<PaginatedResponse<JobFeedItem>> {
  const response = await apiClient.get<PaginatedResponse<JobFeedItem>>("/search/feed", { params: filters });
  return response.data;
}

export async function getHomeSummary(): Promise<JobSearchHomeSummary> {
  const response = await apiClient.get<BaseResponse<JobSearchHomeSummary>>("/search/home-summary");
  return response.data.data as JobSearchHomeSummary;
}

export async function getWhyScore(jobId: string): Promise<WhyThisScorePanel> {
  const response = await apiClient.get<BaseResponse<WhyThisScorePanel>>(`/search/jobs/${jobId}/why-score`);
  return response.data.data as WhyThisScorePanel;
}

export async function getJobDetail(jobId: string): Promise<JobFeedItem> {
  const response = await apiClient.get<BaseResponse<JobFeedItem>>(`/search/jobs/${jobId}`);
  return response.data.data as JobFeedItem;
}

export async function saveJob(jobId: string): Promise<SavedJob> {
  const response = await apiClient.post<BaseResponse<SavedJob>>(`/search/saved-jobs/${jobId}`);
  return response.data.data as SavedJob;
}

export async function unsaveJob(jobId: string): Promise<void> {
  await apiClient.delete(`/search/saved-jobs/${jobId}`);
}

export async function listSavedJobs(page = 1, pageSize = 20): Promise<PaginatedResponse<JobFeedItem>> {
  const response = await apiClient.get<PaginatedResponse<JobFeedItem>>("/search/saved-jobs", {
    params: { page, page_size: pageSize },
  });
  return response.data;
}
