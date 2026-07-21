import { apiClient } from "@/api/client";
import type {
  ApplicantProfileRead,
  ApplicantProfileUpsertRequest,
  AutoApplyAttemptDetail,
  AutoApplyAttemptSummary,
  AutoApplyTaskCreateRequest,
  AutoApplyTaskDetail,
  AutoApplyTaskFiltersParams,
  AutoApplyTaskRead,
} from "@/features/auto-apply/types";
import type { BaseResponse, PaginatedResponse } from "@/types/common";

/** Auto Apply feature — API bindings, `X-User-Id` trusted header convention. */
function authHeaders(userId: string) {
  return { headers: { "X-User-Id": userId } };
}

export async function getApplicantProfile(userId: string): Promise<ApplicantProfileRead> {
  const response = await apiClient.get<BaseResponse<ApplicantProfileRead>>("/auto-apply/profile", authHeaders(userId));
  return response.data.data as ApplicantProfileRead;
}

export async function upsertApplicantProfile(
  userId: string,
  payload: ApplicantProfileUpsertRequest,
): Promise<ApplicantProfileRead> {
  const response = await apiClient.put<BaseResponse<ApplicantProfileRead>>(
    "/auto-apply/profile",
    payload,
    authHeaders(userId),
  );
  return response.data.data as ApplicantProfileRead;
}

export async function createAutoApplyTask(
  userId: string,
  payload: AutoApplyTaskCreateRequest,
): Promise<AutoApplyTaskRead> {
  const response = await apiClient.post<BaseResponse<AutoApplyTaskRead>>("/auto-apply/tasks", payload, authHeaders(userId));
  return response.data.data as AutoApplyTaskRead;
}

export async function listAutoApplyTasks(
  userId: string,
  filters: AutoApplyTaskFiltersParams = {},
): Promise<PaginatedResponse<AutoApplyTaskRead>> {
  const response = await apiClient.get<PaginatedResponse<AutoApplyTaskRead>>("/auto-apply/tasks", {
    ...authHeaders(userId),
    params: filters,
  });
  return response.data;
}

export async function getAutoApplyTask(userId: string, taskId: string): Promise<AutoApplyTaskDetail> {
  const response = await apiClient.get<BaseResponse<AutoApplyTaskDetail>>(
    `/auto-apply/tasks/${taskId}`,
    authHeaders(userId),
  );
  return response.data.data as AutoApplyTaskDetail;
}

export async function retryAutoApplyTask(userId: string, taskId: string): Promise<AutoApplyTaskRead> {
  const response = await apiClient.post<BaseResponse<AutoApplyTaskRead>>(
    `/auto-apply/tasks/${taskId}/retry`,
    undefined,
    authHeaders(userId),
  );
  return response.data.data as AutoApplyTaskRead;
}

export async function cancelAutoApplyTask(userId: string, taskId: string): Promise<AutoApplyTaskRead> {
  const response = await apiClient.post<BaseResponse<AutoApplyTaskRead>>(
    `/auto-apply/tasks/${taskId}/cancel`,
    undefined,
    authHeaders(userId),
  );
  return response.data.data as AutoApplyTaskRead;
}

export async function listAutoApplyAttempts(userId: string, taskId: string): Promise<AutoApplyAttemptSummary[]> {
  const response = await apiClient.get<BaseResponse<AutoApplyAttemptSummary[]>>(
    `/auto-apply/tasks/${taskId}/attempts`,
    authHeaders(userId),
  );
  return response.data.data ?? [];
}

export async function getAutoApplyAttempt(
  userId: string,
  taskId: string,
  attemptId: string,
): Promise<AutoApplyAttemptDetail> {
  const response = await apiClient.get<BaseResponse<AutoApplyAttemptDetail>>(
    `/auto-apply/tasks/${taskId}/attempts/${attemptId}`,
    authHeaders(userId),
  );
  return response.data.data as AutoApplyAttemptDetail;
}

export async function getAutoApplyScreenshotUrl(
  userId: string,
  taskId: string,
  attemptId: string,
  screenshotId: string,
): Promise<string> {
  const response = await apiClient.get<BaseResponse<{ download_url: string }>>(
    `/auto-apply/tasks/${taskId}/attempts/${attemptId}/screenshots/${screenshotId}/download`,
    authHeaders(userId),
  );
  return (response.data.data as { download_url: string }).download_url;
}

/**
 * Phase 0 module-connectivity stub, kept only because nothing has removed
 * its last caller yet.
 */
export async function getAutoApplyModuleStatus(): Promise<BaseResponse> {
  const response = await apiClient.get<BaseResponse>("/auto-apply/status");
  return response.data;
}
