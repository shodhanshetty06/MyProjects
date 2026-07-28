import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { getHomeSummary, getJobFeed, saveJob, unsaveJob } from "@/features/search/api";
import type { JobFeedFiltersParams, SearchRealtimeEvent } from "@/features/search/types";
import { useWorkspaceSocket } from "@/features/workspace/useWorkspaceSocket";
import type { WorkspaceRealtimeEvent } from "@/features/workspace/types";

const JOB_FEED_KEY = "job-feed";
const HOME_SUMMARY_KEY = "job-search-home-summary";
const SAVED_JOBS_KEY = "saved-jobs";

const PAGE_SIZE = 20;

/** Virtualized infinite-scroll job feed — same query shape the old `SearchPage.tsx` used, now backed by `/search/feed`. */
export function useJobFeed(filters: JobFeedFiltersParams) {
  return useInfiniteQuery({
    queryKey: [JOB_FEED_KEY, filters],
    queryFn: ({ pageParam }) => getJobFeed({ ...filters, page: pageParam, page_size: PAGE_SIZE }),
    initialPageParam: 1,
    getNextPageParam: (lastPage) =>
      lastPage.pagination.page < lastPage.pagination.total_pages ? lastPage.pagination.page + 1 : undefined,
  });
}

/** Backs `TopSummaryBar` — greeting, counts, and the recommended next action. */
export function useHomeSummary() {
  return useQuery({
    queryKey: [HOME_SUMMARY_KEY],
    queryFn: getHomeSummary,
  });
}

export function useSaveJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => saveJob(jobId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [JOB_FEED_KEY] });
      void queryClient.invalidateQueries({ queryKey: [SAVED_JOBS_KEY] });
    },
  });
}

export function useUnsaveJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => unsaveJob(jobId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [JOB_FEED_KEY] });
      void queryClient.invalidateQueries({ queryKey: [SAVED_JOBS_KEY] });
    },
  });
}

/**
 * Subscribes to the `search:updates:{user_id}` channel, forwarded over the
 * existing `/workspace/ws` socket (no new socket hook — see
 * `useWorkspaceSocket`, the same pattern `usePipelineProgress` uses to back
 * off a poll with a realtime nudge). Invalidates the job feed and home
 * summary whenever new matches land or the feed goes stale, and the saved
 * list on save/unsave events broadcast from another tab/device.
 */
export function useSearchRealtime(userId: string | null): void {
  const queryClient = useQueryClient();

  useWorkspaceSocket(userId, (event: WorkspaceRealtimeEvent) => {
    const searchEvent = event as unknown as SearchRealtimeEvent;
    switch (searchEvent.type) {
      case "new_matching_jobs":
      case "feed_stale":
        void queryClient.invalidateQueries({ queryKey: [JOB_FEED_KEY] });
        void queryClient.invalidateQueries({ queryKey: [HOME_SUMMARY_KEY] });
        break;
      case "job_saved":
      case "job_unsaved":
        void queryClient.invalidateQueries({ queryKey: [SAVED_JOBS_KEY] });
        void queryClient.invalidateQueries({ queryKey: [JOB_FEED_KEY] });
        break;
      default:
        break;
    }
  });
}
