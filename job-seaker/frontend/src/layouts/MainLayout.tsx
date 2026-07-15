import { Outlet, useNavigate } from "react-router-dom";

import { useAuth } from "@/features/auth";

export function MainLayout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  async function handleLogout() {
    await logout();
    navigate("/auth", { replace: true });
  }

  return (
    <div className="app-shell">
      {user && (
        <header className="app-nav">
          <span>{user.email}</span>
          <button type="button" onClick={handleLogout}>
            Log out
          </button>
        </header>
      )}
      <main className="app-content">
        <Outlet />
      </main>
    </div>
  );
}
