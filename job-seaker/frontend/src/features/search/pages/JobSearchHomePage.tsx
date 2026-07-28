import { useVirtualizer } from "@tanstack/react-virtual";
import { Loader2, SearchX } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { PageHeader } from "@/components/layout/PageHeader";
import { EmptyState } from "@/components/ui/EmptyState";
import { ErrorState } from "@/components/ui/ErrorState";
import { Select } from "@/components/ui/Select";
import { SkeletonCard } from "@/components/ui/Skeleton";
import { TabsList, TabsTrigger, Tabs as UITabs } from "@/components/ui/Tabs";
import { useAuth } from "@/features/auth";
import { AIInsightsStrip } from "@/features/search/components/AIInsightsStrip";
import { FilterPanel } from "@/features/search/components/FilterPanel";
import { JobCard } from "@/features/search/components/JobCard";
import { SearchBar } from "@/features/search/components/SearchBar";
import { TopSummaryBar } from "@/features/search/components/TopSummaryBar";
import { useJobFeed, useSearchRealtime } from "@/features/search/hooks";
import {
  JOB_FEED_MODES,
  type JobFeedFiltersParams,
  type JobFeedMode,
  type ParsedSearchQuery,
  type SearchSortBy,
} from "@/features/search/types";

const SORT_OPTIONS: { value: SearchSortBy; label: string }[] = [
  { value: "relevance", label: "Most relevant" },
  { value: "newest", label: "Newest" },
  { value: "salary", label: "Highest salary" },
  { value: "company", label: "Company A-Z" },
];

/**
 * Replaces the old `SearchPage.tsx`'s "search to get started" empty state:
 * the feed fires immediately on mount (default tab "Recommended For You"),
 * making Job Search the platform's connected front door rather than a
 * dead-end that waits for a query.
 */
export function JobSearchHomePage() {
  const { user } = useAuth();
  const [feedMode, setFeedMode] = useState<JobFeedMode>("recommended");
  const [filters, setFilters] = useState<JobFeedFiltersParams>({ mode: "hybrid", sort_by: "relevance" });
  const [queryText, setQueryText] = useState("");
  const parentRef = useRef<HTMLDivElement>(null);

  useSearchRealtime(user?.id ?? null);

  const feedFilters: JobFeedFiltersParams = { ...filters, feed_mode: feedMode };
  const feedQuery = useJobFeed(feedFilters);

  const results = useMemo(() => feedQuery.data?.pages.flatMap((p) => p.data) ?? [], [feedQuery.data]);
  const totalItems = feedQuery.data?.pages[0]?.pagination.total_items ?? 0;

  const rowVirtualizer = useVirtualizer({
    count: results.length + (feedQuery.hasNextPage ? 1 : 0),
    getScrollElement: () => parentRef.current,
    estimateSize: () => 220,
    overscan: 6,
  });

  const virtualItems = rowVirtualizer.getVirtualItems();
  const lastVirtualIndex = virtualItems[virtualItems.length - 1]?.index;

  useEffect(() => {
    if (lastVirtualIndex === undefined) return;
    if (lastVirtualIndex >= results.length - 1 && feedQuery.hasNextPage && !feedQuery.isFetchingNextPage) {
      void feedQuery.fetchNextPage();
    }
  }, [lastVirtualIndex, results.length, feedQuery]);

  function submitSearch(cleanedQuery: string, parsed?: ParsedSearchQuery) {
    setFilters((prev) => ({
      ...prev,
      q: cleanedQuery || undefined,
      work_mode: parsed?.work_mode ? [parsed.work_mode] : prev.work_mode,
      skills: parsed?.skills.length ? parsed.skills : prev.skills,
      salary_min: parsed?.salary_min ?? prev.salary_min,
      salary_max: parsed?.salary_max ?? prev.salary_max,
      salary_currency: parsed?.salary_currency ?? prev.salary_currency,
      experience_level: parsed?.experience_level ? [parsed.experience_level] : prev.experience_level,
      employment_type: parsed?.employment_type ? [parsed.employment_type] : prev.employment_type,
      country: parsed?.country ? [parsed.country] : prev.country,
      city: parsed?.city ? [parsed.city] : prev.city,
    }));
  }

  function updateFilters(next: JobFeedFiltersParams) {
    setFilters(next);
  }

  return (
    <div>
      <PageHeader
        eyebrow="Job search"
        title="Find your next role"
        description="Personalized recommendations, hybrid keyword+semantic search, and AI match explanations — all in one place."
      />

      <div className="mb-4">
        <TopSummaryBar />
      </div>

      <div className="mb-6">
        <SearchBar
          value={queryText}
          onChange={setQueryText}
          onSubmit={submitSearch}
          mode={filters.mode ?? "hybrid"}
          onModeChange={(mode) => setFilters((prev) => ({ ...prev, mode }))}
        />
      </div>

      <UITabs value={feedMode} onValueChange={(v) => setFeedMode(v as JobFeedMode)} className="mb-4">
        <TabsList className="flex-wrap">
          {JOB_FEED_MODES.map((m) => (
            <TabsTrigger key={m.value} value={m.value}>
              {m.label}
            </TabsTrigger>
          ))}
        </TabsList>
      </UITabs>

      <div className="grid gap-6 lg:grid-cols-[240px_1fr]">
        <aside className="hidden lg:block">
          <FilterPanel filters={filters} onChange={updateFilters} />
        </aside>

        <div className="min-w-0 space-y-4">
          <div className="flex items-center justify-between">
            <p className="text-sm text-ink-secondary">
              {feedQuery.isLoading ? "Loading..." : `${totalItems.toLocaleString()} jobs`}
            </p>
            <Select
              value={filters.sort_by ?? "relevance"}
              onValueChange={(v) => setFilters((prev) => ({ ...prev, sort_by: v as SearchSortBy }))}
              options={SORT_OPTIONS}
              className="w-44"
            />
          </div>

          <AIInsightsStrip />

          {feedQuery.isLoading ? (
            <div className="space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <SkeletonCard key={i} />
              ))}
            </div>
          ) : feedQuery.isError ? (
            <ErrorState
              description="Couldn't load jobs. Try again."
              action={
                <button onClick={() => feedQuery.refetch()} className="text-sm font-medium text-brand">
                  Retry
                </button>
              }
            />
          ) : results.length === 0 ? (
            <EmptyState
              icon={SearchX}
              title="No jobs found"
              description="Try broadening your filters, switching tabs, or adjusting your search."
            />
          ) : (
            <div ref={parentRef} className="h-[calc(100vh-420px)] min-h-[400px] overflow-y-auto">
              <div style={{ height: rowVirtualizer.getTotalSize(), position: "relative" }}>
                {virtualItems.map((virtualRow) => {
                  const job = results[virtualRow.index];
                  return (
                    <div
                      key={virtualRow.key}
                      data-index={virtualRow.index}
                      ref={rowVirtualizer.measureElement}
                      style={{
                        position: "absolute",
                        top: 0,
                        left: 0,
                        width: "100%",
                        transform: `translateY(${virtualRow.start}px)`,
                      }}
                      className="pb-3"
                    >
                      {job ? (
                        <JobCard job={job} />
                      ) : (
                        <div className="flex items-center justify-center py-6 text-ink-muted">
                          <Loader2 className="size-4 animate-spin" aria-hidden />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
