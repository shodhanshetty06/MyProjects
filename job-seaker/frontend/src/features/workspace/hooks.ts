import { useQuery, useQueryClient, type UseQueryResult } from "@tanstack/react-query";

import { getPipelineProgress } from "@/features/workspace/api";
import { useWorkspaceSocket } from "@/features/workspace/useWorkspaceSocket";
import type { PipelineStatus } from "@/features/workspace/types";

const IN_PROGRESS_STATUSES = new Set(["pending", "running"]);

/**
 * Backs the onboarding wizard's single progress screen (and the workspace
 * page's pipeline banner): polls `GET /workspace/progress` while a run is
 * in flight, and refetches immediately whenever the workspace WebSocket
 * reports a `pipeline_progress` event — the poll is the fallback, the
 * socket is what makes it feel instant.
 */
export function usePipelineProgress(userId: string | null): UseQueryResult<PipelineStatus | null> {
  const queryClient = useQueryClient();

  const query = useQuery({
    queryKey: ["pipeline-progress", userId],
    queryFn: getPipelineProgress,
    enabled: !!userId,
    refetchInterval: (q) => {
      const data = q.state.data;
      return !data || IN_PROGRESS_STATUSES.has(data.status) ? 2500 : false;
    },
  });

  useWorkspaceSocket(userId, (event) => {
    if (event.type === "pipeline_progress") {
      void queryClient.invalidateQueries({ queryKey: ["pipeline-progress", userId] });
    }
  });

  return query;
}
