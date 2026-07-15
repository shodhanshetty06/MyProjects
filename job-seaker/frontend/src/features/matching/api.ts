import { apiClient } from "@/api/client";
import { config } from "@/app/config";
import type { BaseResponse, PaginatedResponse } from "@/types/common";

import type {
  MatchPreference,
  MatchPreferenceInput,
  Recommendation,
  RecommendationType,
  ResumeJobMatch,
} from "@/features/matching/types";

/**
 * Matching feature — API bindings.
 *
 * `auth` has no real session yet (Phase 0 stub — see
 * backend/app/api/deps.py::get_current_user_id), so every call here takes an
 * explicit `userId` and sends it as the same trusted `X-User-Id` header the
 * backend expects, rather than inventing a client-side auth layer that
 * doesn't exist anywhere else in this frontend yet.
 */
function authHeaders(userId: string) {
  return { headers: { "X-User-Id": userId } };
}

export async function getPreferences(userId: string): Promise<MatchPreference> {
  const response = await apiClient.get<BaseResponse<MatchPreference>>(
    "/matching/preferences",
    authHeaders(userId),
  );
  return response.data.data as MatchPreference;
}

export async function upsertPreferences(
  userId: string,
  payload: MatchPreferenceInput,
): Promise<MatchPreference> {
  const response = await apiClient.put<BaseResponse<MatchPreference>>(
    "/matching/preferences",
    payload,
    authHeaders(userId),
  );
  return response.data.data as MatchPreference;
}

export async function computeBatchMatches(userId: string, resumeVersionId: string): Promise<void> {
  await apiClient.post(
    `/matching/resumes/${resumeVersionId}/compute`,
    undefined,
    authHeaders(userId),
  );
}

export async function computeSingleMatch(
  userId: string,
  resumeVersionId: string,
  jobId: string,
): Promise<ResumeJobMatch> {
  const response = await apiClient.post<BaseResponse<ResumeJobMatch>>(
    `/matching/resumes/${resumeVersionId}/jobs/${jobId}/compute`,
    undefined,
    authHeaders(userId),
  );
  return response.data.data as ResumeJobMatch;
}

export async function listResumeMatches(
  userId: string,
  resumeVersionId: string,
  page = 1,
  pageSize = 20,
): Promise<PaginatedResponse<ResumeJobMatch>> {
  const response = await apiClient.get<PaginatedResponse<ResumeJobMatch>>(
    `/matching/resumes/${resumeVersionId}/matches`,
    { ...authHeaders(userId), params: { page, page_size: pageSize } },
  );
  return response.data;
}

export async function listRecommendations(
  userId: string,
  type: RecommendationType,
): Promise<Recommendation[]> {
  const response = await apiClient.get<BaseResponse<Recommendation[]>>(
    `/matching/recommendations/${type}s`,
    authHeaders(userId),
  );
  return response.data.data ?? [];
}

export async function dismissRecommendation(
  userId: string,
  recommendationId: string,
): Promise<void> {
  await apiClient.post(
    `/matching/recommendations/${recommendationId}/dismiss`,
    undefined,
    authHeaders(userId),
  );
}

export async function refreshRecommendations(userId: string): Promise<void> {
  await apiClient.post("/matching/recommendations/refresh", undefined, authHeaders(userId));
}

/** `ws(s)://<api-host>/matching/ws?user_id=<id>` — see matching.router's WebSocket endpoint. */
export function buildMatchingSocketUrl(userId: string): string {
  const wsBase = config.apiBaseUrl.replace(/^http/, "ws");
  return `${wsBase}/matching/ws?user_id=${encodeURIComponent(userId)}`;
}
