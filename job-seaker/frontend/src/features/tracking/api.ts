import { apiClient } from "@/api/client";
import type {
  ApplicationCreateRequest,
  ApplicationDetail,
  ApplicationFiltersParams,
  ApplicationNoteRead,
  ApplicationRead,
  ApplicationStatistics,
  ApplicationStatusChangeRequest,
  ApplicationStatusHistoryRead,
  ApplicationUpdateRequest,
  DashboardResponse,
  TimelineEntry,
} from "@/features/tracking/types";
import type { BaseResponse, PaginatedResponse } from "@/types/common";

/**
 * Tracking feature — API bindings. Follows the same `X-User-Id` trusted
 * header convention as `features/matching/api.ts` and
 * `features/ats-resume-generator/api.ts`.
 */
function authHeaders(userId: string) {
  return { headers: { "X-User-Id": userId } };
}

export async function createApplication(
  userId: string,
  payload: ApplicationCreateRequest,
): Promise<ApplicationDetail> {
  const response = await apiClient.post<BaseResponse<ApplicationDetail>>(
    "/tracking/applications",
    payload,
    authHeaders(userId),
  );
  return response.data.data as ApplicationDetail;
}

export async function listApplications(
  userId: string,
  filters: ApplicationFiltersParams = {},
): Promise<PaginatedResponse<ApplicationRead>> {
  const response = await apiClient.get<PaginatedResponse<ApplicationRead>>("/tracking/applications", {
    ...authHeaders(userId),
    params: filters,
  });
  return response.data;
}

export async function getApplication(userId: string, applicationId: string): Promise<ApplicationDetail> {
  const response = await apiClient.get<BaseResponse<ApplicationDetail>>(
    `/tracking/applications/${applicationId}`,
    authHeaders(userId),
  );
  return response.data.data as ApplicationDetail;
}

export async function updateApplication(
  userId: string,
  applicationId: string,
  payload: ApplicationUpdateRequest,
): Promise<ApplicationDetail> {
  const response = await apiClient.patch<BaseResponse<ApplicationDetail>>(
    `/tracking/applications/${applicationId}`,
    payload,
    authHeaders(userId),
  );
  return response.data.data as ApplicationDetail;
}

export async function deleteApplication(userId: string, applicationId: string): Promise<void> {
  await apiClient.delete(`/tracking/applications/${applicationId}`, authHeaders(userId));
}

export async function changeApplicationStatus(
  userId: string,
  applicationId: string,
  payload: ApplicationStatusChangeRequest,
): Promise<ApplicationDetail> {
  const response = await apiClient.post<BaseResponse<ApplicationDetail>>(
    `/tracking/applications/${applicationId}/status`,
    payload,
    authHeaders(userId),
  );
  return response.data.data as ApplicationDetail;
}

export async function getApplicationStatusHistory(
  userId: string,
  applicationId: string,
): Promise<ApplicationStatusHistoryRead[]> {
  const response = await apiClient.get<BaseResponse<ApplicationStatusHistoryRead[]>>(
    `/tracking/applications/${applicationId}/status-history`,
    authHeaders(userId),
  );
  return response.data.data ?? [];
}

export async function getApplicationTimeline(userId: string, applicationId: string): Promise<TimelineEntry[]> {
  const response = await apiClient.get<BaseResponse<TimelineEntry[]>>(
    `/tracking/applications/${applicationId}/timeline`,
    authHeaders(userId),
  );
  return response.data.data ?? [];
}

export async function createApplicationNote(
  userId: string,
  applicationId: string,
  content: string,
): Promise<ApplicationNoteRead> {
  const response = await apiClient.post<BaseResponse<ApplicationNoteRead>>(
    `/tracking/applications/${applicationId}/notes`,
    { content },
    authHeaders(userId),
  );
  return response.data.data as ApplicationNoteRead;
}

export async function listApplicationNotes(userId: string, applicationId: string): Promise<ApplicationNoteRead[]> {
  const response = await apiClient.get<BaseResponse<ApplicationNoteRead[]>>(
    `/tracking/applications/${applicationId}/notes`,
    authHeaders(userId),
  );
  return response.data.data ?? [];
}

export async function updateApplicationNote(
  userId: string,
  applicationId: string,
  noteId: string,
  content: string,
): Promise<ApplicationNoteRead> {
  const response = await apiClient.patch<BaseResponse<ApplicationNoteRead>>(
    `/tracking/applications/${applicationId}/notes/${noteId}`,
    { content },
    authHeaders(userId),
  );
  return response.data.data as ApplicationNoteRead;
}

export async function deleteApplicationNote(userId: string, applicationId: string, noteId: string): Promise<void> {
  await apiClient.delete(`/tracking/applications/${applicationId}/notes/${noteId}`, authHeaders(userId));
}

export async function getApplicationStatistics(userId: string): Promise<ApplicationStatistics> {
  const response = await apiClient.get<BaseResponse<ApplicationStatistics>>(
    "/tracking/statistics",
    authHeaders(userId),
  );
  return response.data.data as ApplicationStatistics;
}

export async function getTrackingDashboard(userId: string): Promise<DashboardResponse> {
  const response = await apiClient.get<BaseResponse<DashboardResponse>>(
    "/tracking/dashboard",
    authHeaders(userId),
  );
  return response.data.data as DashboardResponse;
}

/**
 * Phase 0 module-connectivity stub, kept only because nothing has removed
 * its last caller yet.
 */
export async function getTrackingModuleStatus(): Promise<BaseResponse> {
  const response = await apiClient.get<BaseResponse>("/tracking/status");
  return response.data;
}
