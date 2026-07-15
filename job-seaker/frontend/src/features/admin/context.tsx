import { createContext, useContext, type ReactNode } from "react";

import type { AdminRole, Permission } from "@/features/admin/types";

interface AdminAuthContextValue {
  role: AdminRole;
  permissions: Permission[];
}

const AdminAuthContext = createContext<AdminAuthContextValue | null>(null);

export function AdminAuthProvider({
  value,
  children,
}: {
  value: AdminAuthContextValue;
  children: ReactNode;
}) {
  return <AdminAuthContext.Provider value={value}>{children}</AdminAuthContext.Provider>;
}

export function useAdminAuth(): AdminAuthContextValue {
  const ctx = useContext(AdminAuthContext);
  if (!ctx) {
    throw new Error("useAdminAuth must be used within <AdminLayout>.");
  }
  return ctx;
}

export function useHasPermission(permission: Permission): boolean {
  const { permissions } = useAdminAuth();
  return permissions.includes(permission);
}
