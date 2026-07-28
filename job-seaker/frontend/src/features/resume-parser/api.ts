import type { AxiosProgressEvent } from "axios";

import { apiClient } from "@/api/client";
import type {
  ParsedResumeData,
  ResumeRead,
  ResumeUpdateRequest,
  ResumeVersionDetail,
  ResumeVersionSummary,
} from "@/features/resume-parser/types";
import type { BaseResponse, PaginatedResponse } from "@/types/common";

/**
 * Resume Parser feature — API bindings. Follows the same `X-User-Id`
 * trusted header convention as the other real feature modules.
 */
function authHeaders(userId: string, extra?: Record<string, string | undefined>) {
  return { headers: { "X-User-Id": userId, ...extra } };
}

export async function uploadResume(
  userId: string,
  file: File,
  title?: string,
  onUploadProgress?: (event: AxiosProgressEvent) => void,
): Promise<ResumeRead> {
  const formData = new FormData();
  formData.append("file", file);
  if (title) formData.append("title", title);

  const response = await apiClient.post<BaseResponse<ResumeRead>>("/resumes", formData, {
    // Clears the client's default `Content-Type: application/json` so the
    // browser can set `multipart/form-data` with the correct boundary itself.
    headers: { "X-User-Id": userId, "Content-Type": undefined },
    onUploadProgress,
  });
  return response.data.data as ResumeRead;
}

export async function listResumes(userId: string, page = 1, pageSize = 20): Promise<PaginatedResponse<ResumeRead>> {
  const response = await apiClient.get<PaginatedResponse<ResumeRead>>("/resumes", {
    ...authHeaders(userId),
    params: { page, page_size: pageSize },
  });
  return response.data;
}

export async function getResume(userId: string, resumeId: string): Promise<ResumeRead> {
  const response = await apiClient.get<BaseResponse<ResumeRead>>(`/resumes/${resumeId}`, authHeaders(userId));
  return response.data.data as ResumeRead;
}

export async function updateResume(
  userId: string,
  resumeId: string,
  payload: ResumeUpdateRequest,
): Promise<ResumeRead> {
  const response = await apiClient.patch<BaseResponse<ResumeRead>>(
    `/resumes/${resumeId}`,
    payload,
    authHeaders(userId),
  );
  return response.data.data as ResumeRead;
}

export async function deleteResume(userId: string, resumeId: string): Promise<void> {
  await apiClient.delete(`/resumes/${resumeId}`, authHeaders(userId));
}

export async function uploadResumeVersion(
  userId: string,
  resumeId: string,
  file: File,
  onUploadProgress?: (event: AxiosProgressEvent) => void,
): Promise<ResumeVersionSummary> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await apiClient.post<BaseResponse<ResumeVersionSummary>>(
    `/resumes/${resumeId}/versions`,
    formData,
    { headers: { "X-User-Id": userId, "Content-Type": undefined }, onUploadProgress },
  );
  return response.data.data as ResumeVersionSummary;
}

export async function listResumeVersions(userId: string, resumeId: string): Promise<ResumeVersionSummary[]> {
  const response = await apiClient.get<BaseResponse<ResumeVersionSummary[]>>(
    `/resumes/${resumeId}/versions`,
    authHeaders(userId),
  );
  return response.data.data ?? [];
}

export async function getResumeVersion(
  userId: string,
  resumeId: string,
  versionId: string,
): Promise<ResumeVersionDetail> {
  const response = await apiClient.get<BaseResponse<ResumeVersionDetail>>(
    `/resumes/${resumeId}/versions/${versionId}`,
    authHeaders(userId),
  );
  return response.data.data as ResumeVersionDetail;
}

export async function updateResumeParsedData(
  userId: string,
  resumeId: string,
  versionId: string,
  payload: ParsedResumeData,
): Promise<ResumeVersionDetail> {
  const response = await apiClient.patch<BaseResponse<ResumeVersionDetail>>(
    `/resumes/${resumeId}/versions/${versionId}/parsed-data`,
    payload,
    authHeaders(userId),
  );
  return response.data.data as ResumeVersionDetail;
}

export async function activateResumeVersion(
  userId: string,
  resumeId: string,
  versionId: string,
): Promise<void> {
  await apiClient.put(`/resumes/${resumeId}/versions/${versionId}/activate`, undefined, authHeaders(userId));
}

export async function deleteResumeVersion(userId: string, resumeId: string, versionId: string): Promise<void> {
  await apiClient.delete(`/resumes/${resumeId}/versions/${versionId}`, authHeaders(userId));
}

export async function getResumeVersionDownloadUrl(
  userId: string,
  resumeId: string,
  versionId: string,
): Promise<string> {
  const response = await apiClient.get<BaseResponse<{ download_url: string }>>(
    `/resumes/${resumeId}/versions/${versionId}/download`,
    authHeaders(userId),
  );
  return (response.data.data as { download_url: string }).download_url;
}

/**
 * Phase 0 module-connectivity stub, kept only because nothing has removed
 * its last caller yet.
 */
export async function getResumeParserModuleStatus(): Promise<BaseResponse> {
  const response = await apiClient.get<BaseResponse>("/resume-parser/status");
  return response.data;
}
