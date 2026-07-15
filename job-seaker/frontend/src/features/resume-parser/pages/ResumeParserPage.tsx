import { useQuery } from "@tanstack/react-query";

import { getResumeParserModuleStatus } from "@/features/resume-parser/api";

/**
 * Resume Parser feature — placeholder page.
 *
 * Phase 0: renders the module's wiring status only. Real UI is added once
 * the Resume Parser feature is implemented.
 */
export function ResumeParserPage() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["resume-parser", "status"],
    queryFn: getResumeParserModuleStatus,
  });

  return (
    <section>
      <h1>Resume Parser</h1>
      {isLoading && <p>Checking module status…</p>}
      {isError && <p>Unable to reach the Resume Parser module.</p>}
      {data && <p>{data.message}</p>}
    </section>
  );
}
