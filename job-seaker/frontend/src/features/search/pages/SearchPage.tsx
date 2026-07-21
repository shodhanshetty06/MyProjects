import { useInfiniteQuery } from "@tanstack/react-query";
import { useVirtualizer } from "@tanstack/react-virtual";
import { Loader2, SearchX, SlidersHorizontal } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { PageHeader } from "@/components/layout/PageHeader";
import { EmptyState } from "@/components/ui/EmptyState";
import { ErrorState } from "@/components/ui/ErrorState";
import { Select } from "@/components/ui/Select";
import { SkeletonCard } from "@/components/ui/Skeleton";
import { searchJobs } from "@/features/search/api";
import { FilterPanel } from "@/features/search/components/FilterPanel";
import { JobCard } from "@/features/search/components/JobCard";
import { SearchBar } from "@/features/search/components/SearchBar";
import { TrendingPanel } from "@/features/search/components/TrendingPanel";
import type { SearchFiltersParams, SearchSortBy } from "@/features/search/types";

const PAGE_SIZE = 20;

const SORT_OPTIONS: { value: SearchSortBy; label: string }[] = [
  { value: "relevance", label: "Most relevant" },
  { value: "newest", label: "Newest" },
  { value: "salary", label: "Highest salary" },
  { value: "company", label: "Company A-Z" },
];

export function SearchPage() {
  const [filters, setFilters] = useState<SearchFiltersParams>({ mode: "hybrid", sort_by: "relevance" });
  const [queryText, setQueryText] = useState("");
  const parentRef = useRef<HTMLDivElement>(null);

  const hasQuery = (filters.q ?? "").trim().length > 0;

  const searchQuery = useInfiniteQuery({
    queryKey: ["search", filters],
    queryFn: ({ pageParam }) => searchJobs({ ...filters, page: pageParam, page_size: PAGE_SIZE }),
    initialPageParam: 1,
    getNextPageParam: (lastPage) =>
      lastPage.pagination.page < lastPage.pagination.total_pages ? lastPage.pagination.page + 1 : undefined,
    enabled: hasQuery,
  });

  const results = useMemo(() => searchQuery.data?.pages.flatMap((p) => p.data) ?? [], [searchQuery.data]);
  const totalItems = searchQuery.data?.pages[0]?.pagination.total_items ?? 0;

  const rowVirtualizer = useVirtualizer({
    count: results.length + (searchQuery.hasNextPage ? 1 : 0),
    getScrollElement: () => parentRef.current,
    estimateSize: () => 168,
    overscan: 6,
  });

  const virtualItems = rowVirtualizer.getVirtualItems();
  const lastVirtualIndex = virtualItems[virtualItems.length - 1]?.index;

  useEffect(() => {
    if (lastVirtualIndex === undefined) return;
    if (lastVirtualIndex >= results.length - 1 && searchQuery.hasNextPage && !searchQuery.isFetchingNextPage) {
      void searchQuery.fetchNextPage();
    }
  }, [lastVirtualIndex, results.length, searchQuery]);

  function submitSearch(q: string) {
    setFilters((prev) => ({ ...prev, q: q || undefined }));
  }

  return (
    <div>
      <PageHeader
        eyebrow="Job search"
        title="Find your next role"
        description="Keyword, semantic, or hybrid search across every posted job."
      />

      <div className="mb-6">
        <SearchBar
          value={queryText}
          onChange={setQueryText}
          onSubmit={submitSearch}
          mode={filters.mode ?? "hybrid"}
          onModeChange={(mode) => setFilters((prev) => ({ ...prev, mode }))}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-[240px_1fr]">
        <aside className="hidden lg:block">
          <FilterPanel filters={filters} onChange={setFilters} />
        </aside>

        <div className="min-w-0">
          {!hasQuery ? (
            <div className="grid gap-6 sm:grid-cols-[1fr_320px]">
              <EmptyState
                icon={SlidersHorizontal}
                title="Search for a role to get started"
                description="Try a job title, a skill, or describe the kind of role you want - semantic search understands intent, not just keywords."
                className="h-full"
              />
              <TrendingPanel
                onSkillClick={(skill) => {
                  setQueryText(skill);
                  submitSearch(skill);
                }}
              />
            </div>
          ) : (
            <>
              <div className="mb-3 flex items-center justify-between">
                <p className="text-sm text-ink-secondary">
                  {searchQuery.isLoading ? "Searching..." : `${totalItems.toLocaleString()} results`}
                </p>
                <Select
                  value={filters.sort_by ?? "relevance"}
                  onValueChange={(v) => setFilters((prev) => ({ ...prev, sort_by: v as SearchSortBy }))}
                  options={SORT_OPTIONS}
                  className="w-44"
                />
              </div>

              {searchQuery.isLoading ? (
                <div className="space-y-3">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <SkeletonCard key={i} />
                  ))}
                </div>
              ) : searchQuery.isError ? (
                <ErrorState
                  description="Couldn't load search results. Try again."
                  action={
                    <button onClick={() => searchQuery.refetch()} className="text-sm font-medium text-brand">
                      Retry
                    </button>
                  }
                />
              ) : results.length === 0 ? (
                <EmptyState icon={SearchX} title="No jobs matched" description="Try broadening your filters or switching search modes." />
              ) : (
                <div ref={parentRef} className="h-[calc(100vh-320px)] min-h-[400px] overflow-y-auto">
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
            </>
          )}
        </div>
      </div>
    </div>
  );
}
