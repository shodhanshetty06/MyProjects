import { apiClient } from "@/api/client";
import type { BaseResponse } from "@/types/common";

/**
 * Resume Parser feature — API bindings.
 *
 * Phase 0: only the module status check is wired up, to verify the
 * frontend can reach the corresponding backend module scaffold.
 */
export async function getResumeParserModuleStatus(): Promise<BaseResponse> {
  const response = await apiClient.get<BaseResponse>("/resume-parser/status");
  return response.data;
}
