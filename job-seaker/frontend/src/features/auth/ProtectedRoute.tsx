import { Navigate, Outlet, useLocation } from "react-router-dom";

import { useAuth } from "@/features/auth/AuthProvider";

/** Gates every module route behind a logged-in session, redirecting to
 * `/auth` (and back again on success) otherwise. */
export function ProtectedRoute() {
  const { isAuthenticated, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) {
    return <p>Loading…</p>;
  }
  if (!isAuthenticated) {
    return <Navigate to="/auth" replace state={{ from: location }} />;
  }
  return <Outlet />;
}
