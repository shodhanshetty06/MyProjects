import { apiClient } from "@/api/client";
import type {
  AdminActorBootstrapRequest,
  AdminActorCreateRequest,
  AdminActorResponse,
  AdminActorUpdateRequest,
  ApplicationFiltersParams,
  ApplicationSummary,
  AuditLogEntry,
  AuditLogFiltersParams,
  CompanyDetail,
  CompanyFiltersParams,
  CompanySummary,
  CompanyVerificationUpdateRequest,
  DashboardResponse,
  FeatureFlagCreateRequest,
  FeatureFlagResponse,
  FeatureFlagUpdateRequest,
  JobFiltersParams,
  JobModerationUpdateRequest,
  JobSummary,
  ModerationQueueResponse,
  PermissionsResponse,
  PlatformAnalyticsResponse,
  ReportCreateRequest,
  ReportFiltersParams,
  ReportResponse,
  ReportUpdateRequest,
  SystemLogEntry,
  SystemLogFiltersParams,
  UserDetail,
  UserFiltersParams,
  UserStatusUpdateRequest,
  UserSummary,
} from "@/features/admin/types";
import type { BaseResponse, PaginatedResponse } from "@/types/common";

/**
 * Admin feature — API bindings.
 *
 * Mirrors backend/app/modules/admin/router.py. `unwrap`/`unwrapPage` throw on
 * a missing `data` field so callers can treat every function as returning
 * the real payload, matching the `analytics` feature's `api.ts` convention.
 */

function unwrap<T>(response: BaseResponse<T>): T {
  if (response.data === null || response.data === undefined) {
    throw new Error(response.message || "Admin API response had no data.");
  }
  return response.data;
}

interface Page<T> {
  items: T[];
  page: number;
  pageSize: number;
  totalItems: number;
  totalPages: number;
}

function unwrapPage<T>(response: PaginatedResponse<T>): Page<T> {
  return {
    items: response.data,
    page: response.pagination.page,
    pageSize: response.pagination.page_size,
    totalItems: response.pagination.total_items,
    totalPages: response.pagination.total_pages,
  };
}

// --- actors / permissions --------------------------------------------------

export async function bootstrapAdminActor(
  request: AdminActorBootstrapRequest,
): Promise<AdminActorResponse> {
  const response = await apiClient.post<BaseResponse<AdminActorResponse>>(
    "/admin/actors/bootstrap",
    request,
  );
  return unwrap(response.data);
}

export async function listAdminActors(page = 1, pageSize = 20): Promise<Page<AdminActorResponse>> {
  const response = await apiClient.get<PaginatedResponse<AdminActorResponse>>("/admin/actors", {
    params: { page, page_size: pageSize },
  });
  return unwrapPage(response.data);
}

export async function createAdminActor(
  request: AdminActorCreateRequest,
): Promise<AdminActorResponse> {
  const response = await apiClient.post<BaseResponse<AdminActorResponse>>("/admin/actors", request);
  return unwrap(response.data);
}

export async function updateAdminActor(
  actorId: string,
  request: AdminActorUpdateRequest,
): Promise<AdminActorResponse> {
  const response = await apiClient.patch<BaseResponse<AdminActorResponse>>(
    `/admin/actors/${actorId}`,
    request,
  );
  return unwrap(response.data);
}

export async function getPermissions(): Promise<PermissionsResponse> {
  const response = await apiClient.get<BaseResponse<PermissionsResponse>>("/admin/permissions");
  return unwrap(response.data);
}

// --- dashboard ---------------------------------------------------------------

export async function getDashboard(): Promise<DashboardResponse> {
  const response = await apiClient.get<BaseResponse<DashboardResponse>>("/admin/dashboard");
  return unwrap(response.data);
}

// --- users ---------------------------------------------------------------

export async function listUsers(filters: UserFiltersParams): Promise<Page<UserSummary>> {
  const response = await apiClient.get<PaginatedResponse<UserSummary>>("/admin/users", {
    params: filters,
  });
  return unwrapPage(response.data);
}

export async function getUser(userId: string): Promise<UserDetail> {
  const response = await apiClient.get<BaseResponse<UserDetail>>(`/admin/users/${userId}`);
  return unwrap(response.data);
}

export async function updateUserStatus(
  userId: string,
  request: UserStatusUpdateRequest,
): Promise<UserDetail> {
  const response = await apiClient.patch<BaseResponse<UserDetail>>(
    `/admin/users/${userId}/status`,
    request,
  );
  return unwrap(response.data);
}

// --- companies --------------------------------------------------------------

export async function listCompanies(filters: CompanyFiltersParams): Promise<Page<CompanySummary>> {
  const response = await apiClient.get<PaginatedResponse<CompanySummary>>("/admin/companies", {
    params: filters,
  });
  return unwrapPage(response.data);
}

export async function getCompany(companyId: string): Promise<CompanyDetail> {
  const response = await apiClient.get<BaseResponse<CompanyDetail>>(`/admin/companies/${companyId}`);
  return unwrap(response.data);
}

export async function updateCompanyVerification(
  companyId: string,
  request: CompanyVerificationUpdateRequest,
): Promise<CompanyDetail> {
  const response = await apiClient.patch<BaseResponse<CompanyDetail>>(
    `/admin/companies/${companyId}/verification`,
    request,
  );
  return unwrap(response.data);
}

// --- jobs ---------------------------------------------------------------

export async function listJobs(filters: JobFiltersParams): Promise<Page<JobSummary>> {
  const response = await apiClient.get<PaginatedResponse<JobSummary>>("/admin/jobs", {
    params: filters,
  });
  return unwrapPage(response.data);
}

export async function updateJobModeration(
  jobId: string,
  request: JobModerationUpdateRequest,
): Promise<JobSummary> {
  const response = await apiClient.patch<BaseResponse<JobSummary>>(
    `/admin/jobs/${jobId}/moderation`,
    request,
  );
  return unwrap(response.data);
}

// --- applications ---------------------------------------------------------

export async function listApplications(
  filters: ApplicationFiltersParams,
): Promise<Page<ApplicationSummary>> {
  const response = await apiClient.get<PaginatedResponse<ApplicationSummary>>(
    "/admin/applications",
    { params: filters },
  );
  return unwrapPage(response.data);
}

export async function deleteApplication(applicationId: string): Promise<void> {
  await apiClient.delete(`/admin/applications/${applicationId}`);
}

// --- reports / content moderation -------------------------------------------

export async function createReport(request: ReportCreateRequest): Promise<ReportResponse> {
  const response = await apiClient.post<BaseResponse<ReportResponse>>("/admin/reports", request);
  return unwrap(response.data);
}

export async function listReports(filters: ReportFiltersParams): Promise<Page<ReportResponse>> {
  const response = await apiClient.get<PaginatedResponse<ReportResponse>>("/admin/reports", {
    params: filters,
  });
  return unwrapPage(response.data);
}

export async function updateReport(
  reportId: string,
  request: ReportUpdateRequest,
): Promise<ReportResponse> {
  const response = await apiClient.patch<BaseResponse<ReportResponse>>(
    `/admin/reports/${reportId}`,
    request,
  );
  return unwrap(response.data);
}

export async function getModerationQueue(): Promise<ModerationQueueResponse> {
  const response = await apiClient.get<BaseResponse<ModerationQueueResponse>>(
    "/admin/moderation/queue",
  );
  return unwrap(response.data);
}

// --- feature flags ------------------------------------------------------------

export async function listFeatureFlags(): Promise<FeatureFlagResponse[]> {
  const response = await apiClient.get<BaseResponse<FeatureFlagResponse[]>>("/admin/feature-flags");
  return unwrap(response.data);
}

export async function createFeatureFlag(
  request: FeatureFlagCreateRequest,
): Promise<FeatureFlagResponse> {
  const response = await apiClient.post<BaseResponse<FeatureFlagResponse>>(
    "/admin/feature-flags",
    request,
  );
  return unwrap(response.data);
}

export async function updateFeatureFlag(
  key: string,
  request: FeatureFlagUpdateRequest,
): Promise<FeatureFlagResponse> {
  const response = await apiClient.patch<BaseResponse<FeatureFlagResponse>>(
    `/admin/feature-flags/${key}`,
    request,
  );
  return unwrap(response.data);
}

export async function deleteFeatureFlag(key: string): Promise<void> {
  await apiClient.delete(`/admin/feature-flags/${key}`);
}

// --- audit logs / system logs ------------------------------------------------

export async function listAuditLogs(filters: AuditLogFiltersParams): Promise<Page<AuditLogEntry>> {
  const response = await apiClient.get<PaginatedResponse<AuditLogEntry>>("/admin/audit-logs", {
    params: filters,
  });
  return unwrapPage(response.data);
}

export async function listSystemLogs(
  filters: SystemLogFiltersParams,
): Promise<Page<SystemLogEntry>> {
  const response = await apiClient.get<PaginatedResponse<SystemLogEntry>>("/admin/system-logs", {
    params: filters,
  });
  return unwrapPage(response.data);
}

// --- platform analytics --------------------------------------------------------

export async function getPlatformAnalytics(days = 30): Promise<PlatformAnalyticsResponse> {
  const response = await apiClient.get<BaseResponse<PlatformAnalyticsResponse>>("/admin/analytics", {
    params: { days },
  });
  return unwrap(response.data);
}

export type { Page };
