import { useQuery } from "@tanstack/react-query";

import { getJobIngestionModuleStatus } from "@/features/job-ingestion/api";

/**
 * Job Ingestion feature — placeholder page.
 *
 * Phase 0: renders the module's wiring status only. Real UI is added once
 * the Job Ingestion feature is implemented.
 */
export function JobIngestionPage() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["job-ingestion", "status"],
    queryFn: getJobIngestionModuleStatus,
  });

  return (
    <section>
      <h1>Job Ingestion</h1>
      {isLoading && <p>Checking module status…</p>}
      {isError && <p>Unable to reach the Job Ingestion module.</p>}
      {data && <p>{data.message}</p>}
    </section>
  );
}
