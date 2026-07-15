import { useQuery } from "@tanstack/react-query";

import { getNotificationModuleStatus } from "@/features/notification/api";

/**
 * Notification feature — placeholder page.
 *
 * Phase 0: renders the module's wiring status only. Real UI is added once
 * the Notification feature is implemented.
 */
export function NotificationPage() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["notification", "status"],
    queryFn: getNotificationModuleStatus,
  });

  return (
    <section>
      <h1>Notification</h1>
      {isLoading && <p>Checking module status…</p>}
      {isError && <p>Unable to reach the Notification module.</p>}
      {data && <p>{data.message}</p>}
    </section>
  );
}
