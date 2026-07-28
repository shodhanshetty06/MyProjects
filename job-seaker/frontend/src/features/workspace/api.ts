import { apiClient } from "@/api/client";
import { config } from "@/app/config";
import type { PipelineStatus, WorkspaceSummary } from "@/features/workspace/types";
import type { BaseResponse } from "@/types/common";

export async function getWorkspace(): Promise<WorkspaceSummary> {
  const response = await apiClient.get<BaseResponse<WorkspaceSummary>>("/workspace");
  return response.data.data as WorkspaceSummary;
}

export async function getPipelineProgress(): Promise<PipelineStatus | null> {
  const response = await apiClient.get<BaseResponse<PipelineStatus | null>>("/workspace/progress");
  return response.data.data ?? null;
}

/** `ws(s)://<api-host>/workspace/ws?user_id=<id>` — see workspace.router's WebSocket endpoint. */
export function buildWorkspaceSocketUrl(userId: string): string {
  const wsBase = config.apiBaseUrl.replace(/^http/, "ws");
  return `${wsBase}/workspace/ws?user_id=${encodeURIComponent(userId)}`;
}
