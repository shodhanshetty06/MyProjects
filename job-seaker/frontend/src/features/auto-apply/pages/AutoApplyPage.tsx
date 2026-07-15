import { useQuery } from "@tanstack/react-query";

import { getAutoApplyModuleStatus } from "@/features/auto-apply/api";

/**
 * Auto Apply feature — placeholder page.
 *
 * Phase 0: renders the module's wiring status only. Real UI is added once
 * the Auto Apply feature is implemented.
 */
export function AutoApplyPage() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["auto-apply", "status"],
    queryFn: getAutoApplyModuleStatus,
  });

  return (
    <section>
      <h1>Auto Apply</h1>
      {isLoading && <p>Checking module status…</p>}
      {isError && <p>Unable to reach the Auto Apply module.</p>}
      {data && <p>{data.message}</p>}
    </section>
  );
}
