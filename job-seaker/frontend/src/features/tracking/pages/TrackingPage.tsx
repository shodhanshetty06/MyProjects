import { useQuery } from "@tanstack/react-query";

import { getTrackingModuleStatus } from "@/features/tracking/api";

/**
 * Tracking feature — placeholder page.
 *
 * Phase 0: renders the module's wiring status only. Real UI is added once
 * the Tracking feature is implemented.
 */
export function TrackingPage() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["tracking", "status"],
    queryFn: getTrackingModuleStatus,
  });

  return (
    <section>
      <h1>Tracking</h1>
      {isLoading && <p>Checking module status…</p>}
      {isError && <p>Unable to reach the Tracking module.</p>}
      {data && <p>{data.message}</p>}
    </section>
  );
}
