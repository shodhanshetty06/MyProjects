import { apiClient } from "@/api/client";
import type { BaseResponse } from "@/types/common";

/**
 * Notification feature — API bindings.
 *
 * Phase 0: only the module status check is wired up, to verify the
 * frontend can reach the corresponding backend module scaffold.
 */
export async function getNotificationModuleStatus(): Promise<BaseResponse> {
  const response = await apiClient.get<BaseResponse>("/notifications/status");
  return response.data;
}
