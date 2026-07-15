import { apiClient } from "@/api/client";
import type { BaseResponse } from "@/types/common";

/**
 * Search feature — API bindings.
 *
 * Phase 0: only the module status check is wired up, to verify the
 * frontend can reach the corresponding backend module scaffold.
 */
export async function getSearchModuleStatus(): Promise<BaseResponse> {
  const response = await apiClient.get<BaseResponse>("/search/status");
  return response.data;
}
