import { useQuery } from "@tanstack/react-query";

import { getSearchModuleStatus } from "@/features/search/api";

/**
 * Search feature — placeholder page.
 *
 * Phase 0: renders the module's wiring status only. Real UI is added once
 * the Search feature is implemented.
 */
export function SearchPage() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["search", "status"],
    queryFn: getSearchModuleStatus,
  });

  return (
    <section>
      <h1>Search</h1>
      {isLoading && <p>Checking module status…</p>}
      {isError && <p>Unable to reach the Search module.</p>}
      {data && <p>{data.message}</p>}
    </section>
  );
}
