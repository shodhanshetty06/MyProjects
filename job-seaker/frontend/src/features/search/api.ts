import { apiClient } from "@/api/client";
import type {
  AutocompleteSuggestion,
  SearchFiltersParams,
  SearchResultItem,
  SearchSuggestion,
  TrendingJob,
  TrendingSkill,
} from "@/features/search/types";
import type { BaseResponse, PaginatedResponse } from "@/types/common";

/** Search feature — API bindings. All endpoints are public (no auth header). */
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
