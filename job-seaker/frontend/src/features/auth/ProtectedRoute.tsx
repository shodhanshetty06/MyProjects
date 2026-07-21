import { Navigate, Outlet, useLocation } from "react-router-dom";

import { PageLoader } from "@/components/common/LoadingSpinner";
import { useAuth } from "@/features/auth/AuthProvider";
import { paths } from "@/routes/paths";

/** Gates every module route behind a logged-in session, redirecting to
 * the login page (and back again on success) otherwise. */
export function ProtectedRoute() {
  const { isAuthenticated, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) {
    return <PageLoader />;
  }
  if (!isAuthenticated) {
    return <Navigate to={paths.login} replace state={{ from: location }} />;
  }
  return <Outlet />;
}
