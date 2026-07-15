import { apiClient } from "@/api/client";
import type { BaseResponse } from "@/types/common";

/**
 * Job Ingestion feature — API bindings.
 *
 * Phase 0: only the module status check is wired up, to verify the
 * frontend can reach the corresponding backend module scaffold.
 */
export async function getJobIngestionModuleStatus(): Promise<BaseResponse> {
  const response = await apiClient.get<BaseResponse>("/job-ingestion/status");
  return response.data;
}
