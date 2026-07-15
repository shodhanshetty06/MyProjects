import { useQuery } from "@tanstack/react-query";

import { getUserProfileModuleStatus } from "@/features/user-profile/api";

/**
 * User Profile feature — placeholder page.
 *
 * Phase 0: renders the module's wiring status only. Real UI is added once
 * the User Profile feature is implemented.
 */
export function UserProfilePage() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["user-profile", "status"],
    queryFn: getUserProfileModuleStatus,
  });

  return (
    <section>
      <h1>User Profile</h1>
      {isLoading && <p>Checking module status…</p>}
      {isError && <p>Unable to reach the User Profile module.</p>}
      {data && <p>{data.message}</p>}
    </section>
  );
}
