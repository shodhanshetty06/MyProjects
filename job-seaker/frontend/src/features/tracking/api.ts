import { apiClient } from "@/api/client";
import type { BaseResponse } from "@/types/common";

/**
 * Tracking feature — API bindings.
 *
 * Phase 0: only the module status check is wired up, to verify the
 * frontend can reach the corresponding backend module scaffold.
 */
export async function getTrackingModuleStatus(): Promise<BaseResponse> {
  const response = await apiClient.get<BaseResponse>("/tracking/status");
  return response.data;
}
