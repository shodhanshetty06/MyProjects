import { apiClient } from "@/api/client";
import type { CompanyRead, JobListFiltersParams, JobRead } from "@/features/job-ingestion/types";
import type { BaseResponse, PaginatedResponse } from "@/types/common";

/**
 * Job Ingestion feature — API bindings. This router has no auth
 * dependency on the backend (see backend/app/modules/job_ingestion/router.py),
 * so these calls carry no user header — used here as the public read-only
 * company/job directory behind "Company Explorer" and job detail pages.
 */
export async function listCompanies(page = 1, pageSize = 20): Promise<PaginatedResponse<CompanyRead>> {
  const response = await apiClient.get<PaginatedResponse<CompanyRead>>("/job-ingestion/companies", {
    params: { page, page_size: pageSize },
  });
  return response.data;
}

export async function getCompany(companyId: string): Promise<CompanyRead> {
  const response = await apiClient.get<BaseResponse<CompanyRead>>(`/job-ingestion/companies/${companyId}`);
  return response.data.data as CompanyRead;
}

export async function listJobs(filters: JobListFiltersParams = {}): Promise<PaginatedResponse<JobRead>> {
  const response = await apiClient.get<PaginatedResponse<JobRead>>("/job-ingestion/jobs", { params: filters });
  return response.data;
}

export async function getJob(jobId: string): Promise<JobRead> {
  const response = await apiClient.get<BaseResponse<JobRead>>(`/job-ingestion/jobs/${jobId}`);
  return response.data.data as JobRead;
}

/**
 * Phase 0 module-connectivity stub, kept only because nothing has removed
 * its last caller yet.
 */
export async function getJobIngestionModuleStatus(): Promise<BaseResponse> {
  const response = await apiClient.get<BaseResponse>("/job-ingestion/status");
  return response.data;
}
