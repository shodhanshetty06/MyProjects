import { KeyRound, Shield, User } from "lucide-react";
import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";

import { PageHeader } from "@/components/layout/PageHeader";
import { cn } from "@/lib/cn";
import { paths } from "@/routes/paths";

const NAV = [
  { to: paths.settingsProfile, label: "Profile", icon: User },
  { to: paths.settingsSecurity, label: "Security", icon: Shield },
  { to: paths.settingsApiKeys, label: "API Keys", icon: KeyRound },
];

export function SettingsShell({ children }: { children: ReactNode }) {
  return (
    <div>
      <PageHeader eyebrow="Account" title="Settings" description="Manage your profile, security, and API access." />
      <div className="grid gap-6 lg:grid-cols-[200px_1fr]">
        <nav className="flex gap-1 overflow-x-auto lg:flex-col" aria-label="Settings navigation">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                cn(
                  "flex shrink-0 items-center gap-2 rounded-[var(--radius-md)] px-3 py-2 text-sm font-medium transition-colors",
                  isActive ? "bg-brand-soft text-brand" : "text-ink-secondary hover:bg-surface-2 hover:text-ink",
                )
              }
            >
              <item.icon className="size-4" aria-hidden />
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="min-w-0">{children}</div>
      </div>
    </div>
  );
}
