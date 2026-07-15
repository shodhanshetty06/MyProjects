import { apiClient } from "@/api/client";
import type { BaseResponse, PaginatedResponse } from "@/types/common";

import type {
  ATSFileFormat,
  ATSResumeCreateRequest,
  ATSResumeDownloadResponse,
  ATSResumeRead,
  ATSResumeVersionCreateRequest,
  ATSResumeVersionDetail,
  ATSResumeVersionDiff,
  ATSResumeVersionSummary,
} from "@/features/ats-resume-generator/types";

/**
 * ATS Resume Generator feature — API bindings.
 *
 * `auth` has no real session yet (Phase 0 stub — see
 * backend/app/api/deps.py::get_current_user_id), so every call here takes an
 * explicit `userId` and sends it as the same trusted `X-User-Id` header the
 * backend expects, mirroring `features/matching/api.ts`.
 */
function authHeaders(userId: string) {
  return { headers: { "X-User-Id": userId } };
}

export async function createATSResume(
  userId: string,
  payload: ATSResumeCreateRequest,
): Promise<ATSResumeRead> {
  const response = await apiClient.post<BaseResponse<ATSResumeRead>>(
    "/ats-resumes",
    payload,
    authHeaders(userId),
  );
  return response.data.data as ATSResumeRead;
}

export async function listATSResumes(
  userId: string,
  page = 1,
  pageSize = 20,
): Promise<PaginatedResponse<ATSResumeRead>> {
  const response = await apiClient.get<PaginatedResponse<ATSResumeRead>>("/ats-resumes", {
    ...authHeaders(userId),
    params: { page, page_size: pageSize },
  });
  return response.data;
}

export async function getATSResume(userId: string, atsResumeId: string): Promise<ATSResumeRead> {
  const response = await apiClient.get<BaseResponse<ATSResumeRead>>(
    `/ats-resumes/${atsResumeId}`,
    authHeaders(userId),
  );
  return response.data.data as ATSResumeRead;
}

export async function updateATSResume(
  userId: string,
  atsResumeId: string,
  title: string,
): Promise<ATSResumeRead> {
  const response = await apiClient.patch<BaseResponse<ATSResumeRead>>(
    `/ats-resumes/${atsResumeId}`,
    { title },
    authHeaders(userId),
  );
  return response.data.data as ATSResumeRead;
}

export async function deleteATSResume(userId: string, atsResumeId: string): Promise<void> {
  await apiClient.delete(`/ats-resumes/${atsResumeId}`, authHeaders(userId));
}

export async function createATSResumeVersion(
  userId: string,
  atsResumeId: string,
  payload: ATSResumeVersionCreateRequest,
): Promise<ATSResumeVersionSummary> {
  const response = await apiClient.post<BaseResponse<ATSResumeVersionSummary>>(
    `/ats-resumes/${atsResumeId}/versions`,
    payload,
    authHeaders(userId),
  );
  return response.data.data as ATSResumeVersionSummary;
}

export async function listATSResumeVersions(
  userId: string,
  atsResumeId: string,
): Promise<ATSResumeVersionSummary[]> {
  const response = await apiClient.get<BaseResponse<ATSResumeVersionSummary[]>>(
    `/ats-resumes/${atsResumeId}/versions`,
    authHeaders(userId),
  );
  return response.data.data ?? [];
}

export async function getATSResumeVersion(
  userId: string,
  atsResumeId: string,
  versionId: string,
): Promise<ATSResumeVersionDetail> {
  const response = await apiClient.get<BaseResponse<ATSResumeVersionDetail>>(
    `/ats-resumes/${atsResumeId}/versions/${versionId}`,
    authHeaders(userId),
  );
  return response.data.data as ATSResumeVersionDetail;
}

export async function activateATSResumeVersion(
  userId: string,
  atsResumeId: string,
  versionId: string,
): Promise<ATSResumeVersionSummary> {
  const response = await apiClient.put<BaseResponse<ATSResumeVersionSummary>>(
    `/ats-resumes/${atsResumeId}/versions/${versionId}/activate`,
    undefined,
    authHeaders(userId),
  );
  return response.data.data as ATSResumeVersionSummary;
}

export async function deleteATSResumeVersion(
  userId: string,
  atsResumeId: string,
  versionId: string,
): Promise<void> {
  await apiClient.delete(
    `/ats-resumes/${atsResumeId}/versions/${versionId}`,
    authHeaders(userId),
  );
}

export async function getATSResumeVersionDiff(
  userId: string,
  atsResumeId: string,
  versionId: string,
  compareTo = "source",
): Promise<ATSResumeVersionDiff> {
  const response = await apiClient.get<BaseResponse<ATSResumeVersionDiff>>(
    `/ats-resumes/${atsResumeId}/versions/${versionId}/diff`,
    { ...authHeaders(userId), params: { compare_to: compareTo } },
  );
  return response.data.data as ATSResumeVersionDiff;
}

export async function getATSResumeDownloadUrl(
  userId: string,
  atsResumeId: string,
  versionId: string,
  format: ATSFileFormat,
): Promise<string> {
  const response = await apiClient.get<BaseResponse<ATSResumeDownloadResponse>>(
    `/ats-resumes/${atsResumeId}/versions/${versionId}/download/${format}`,
    authHeaders(userId),
  );
  return (response.data.data as ATSResumeDownloadResponse).download_url;
}
