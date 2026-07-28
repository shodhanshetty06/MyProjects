import {
  BarChart3,
  Bell,
  Bot,
  Briefcase,
  ChevronsLeft,
  ChevronsRight,
  FileUp,
  LayoutDashboard,
  LogOut,
  Menu,
  Search,
  Settings,
  Shield,
  Sparkles,
  User,
  X,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { Suspense, useEffect, useState, type ComponentType } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";

import { CommandPalette, useCommandPaletteHotkey } from "@/components/command/CommandPalette";
import { KeyboardShortcutsDialog } from "@/components/command/KeyboardShortcutsDialog";
import { PageLoader } from "@/components/common/LoadingSpinner";
import { PageTransition } from "@/components/motion/PageTransition";
import { Avatar } from "@/components/ui/Avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/DropdownMenu";
import { Kbd } from "@/components/ui/Kbd";
import { Tooltip } from "@/components/ui/Tooltip";
import { useAuth } from "@/features/auth";
import { listNotifications } from "@/features/notification";
import { cn } from "@/lib/cn";
import { paths } from "@/routes/paths";
import { useUiStore } from "@/stores/uiStore";

interface NavItem {
  to: string;
  label: string;
  icon: ComponentType<{ className?: string }>;
}

/**
 * Flat primary nav — the AI Workspace replaces separate Resume Analysis /
 * ATS Analysis / Manual Recommendations pages as the everyday entry point,
 * so those (and Auto Apply, Companies, Saved/Recommended Jobs standalone,
 * the Notifications page) stay registered in `AppRoutes.tsx` for
 * deep-linking but drop out of this list.
 */
const NAV_ITEMS: NavItem[] = [
  { to: paths.workspace, label: "Workspace", icon: LayoutDashboard },
  { to: paths.search, label: "Jobs", icon: Search },
  { to: paths.tracker, label: "Applications", icon: Briefcase },
  { to: paths.coach, label: "AI Coach", icon: Bot },
  { to: paths.analytics, label: "Analytics", icon: BarChart3 },
  { to: paths.settingsProfile, label: "Settings", icon: Settings },
];

export function AppShell() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const collapsed = useUiStore((state) => state.sidebarCollapsed);
  const toggleSidebar = useUiStore((state) => state.toggleSidebar);
  const setCommandPaletteOpen = useUiStore((state) => state.setCommandPaletteOpen);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const shortcutsOpen = useUiStore((state) => state.shortcutsOpen);
  const setShortcutsOpen = useUiStore((state) => state.setShortcutsOpen);
  useCommandPaletteHotkey();

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      const isTyping = target?.tagName === "INPUT" || target?.tagName === "TEXTAREA" || target?.isContentEditable;
      if (event.key === "?" && !isTyping) {
        event.preventDefault();
        setShortcutsOpen(!shortcutsOpen);
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [shortcutsOpen, setShortcutsOpen]);

  const unreadQuery = useQuery({
    queryKey: ["notifications", user?.id, "unread-count"],
    queryFn: () => listNotifications(user!.id, true, 1, 1),
    enabled: !!user,
    refetchInterval: 45_000,
  });
  const unreadCount = unreadQuery.data?.pagination.total_items ?? 0;

  async function handleLogout() {
    await logout();
    navigate(paths.login, { replace: true });
  }

  if (!user) return <Outlet />;

  return (
    <div className="flex min-h-screen bg-bg">
      {mobileNavOpen && (
        <div
          aria-hidden
          onClick={() => setMobileNavOpen(false)}
          className="fixed inset-0 z-40 bg-overlay backdrop-blur-[1px] lg:hidden"
        />
      )}

      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-50 flex h-screen w-64 shrink-0 flex-col border-r border-border bg-surface",
          "transition-transform duration-200 ease-out lg:sticky lg:top-0 lg:translate-x-0 lg:transition-[width]",
          collapsed ? "lg:w-[72px]" : "lg:w-[236px]",
          mobileNavOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className={cn("flex h-14 items-center gap-2 px-4", collapsed && "lg:justify-center lg:px-0")}>
          <div className="flex size-7 shrink-0 items-center justify-center rounded-[var(--radius-sm)] bg-[linear-gradient(135deg,var(--color-accent-a),var(--color-accent-b))]">
            <Sparkles className="size-4 text-white" aria-hidden />
          </div>
          <span className={cn("text-sm font-semibold text-ink", collapsed && "lg:hidden")}>Ascend AI</span>
          <button
            type="button"
            onClick={() => setMobileNavOpen(false)}
            className="ml-auto flex size-7 items-center justify-center rounded-[var(--radius-sm)] text-ink-muted hover:bg-surface-2 lg:hidden"
            aria-label="Close navigation"
          >
            <X className="size-4" aria-hidden />
          </button>
        </div>

        <nav className="flex flex-1 flex-col gap-0.5 overflow-y-auto px-3 py-3" aria-label="Main navigation">
          {NAV_ITEMS.map((item) => {
            const link = (
              <NavLink
                key={item.to}
                to={item.to}
                onClick={() => setMobileNavOpen(false)}
                className={({ isActive }) =>
                  cn(
                    "flex items-center gap-2.5 rounded-[var(--radius-md)] px-2.5 py-2 text-sm font-medium transition-colors",
                    collapsed && "lg:justify-center lg:px-0",
                    isActive
                      ? "bg-brand-soft text-brand"
                      : "text-ink-secondary hover:bg-surface-2 hover:text-ink",
                  )
                }
              >
                <item.icon className="size-4 shrink-0" />
                <span className={cn(collapsed && "lg:hidden")}>{item.label}</span>
              </NavLink>
            );
            return collapsed ? (
              <Tooltip key={item.to} content={item.label} side="right">
                {link}
              </Tooltip>
            ) : (
              link
            );
          })}
        </nav>

        <div className="flex flex-col gap-1 border-t border-border px-3 py-3">
          <button
            type="button"
            onClick={toggleSidebar}
            className={cn(
              "hidden items-center gap-2.5 rounded-[var(--radius-md)] px-2.5 py-2 text-sm font-medium text-ink-secondary transition-colors hover:bg-surface-2 hover:text-ink lg:flex",
              collapsed && "lg:justify-center lg:px-0",
            )}
          >
            {collapsed ? <ChevronsRight className="size-4 shrink-0" /> : <ChevronsLeft className="size-4 shrink-0" />}
            <span className={cn(collapsed && "lg:hidden")}>Collapse</span>
          </button>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-border bg-surface/80 px-5 backdrop-blur">
          <button
            type="button"
            onClick={() => setMobileNavOpen(true)}
            className="flex size-8 shrink-0 items-center justify-center rounded-[var(--radius-md)] text-ink-secondary hover:bg-surface-2 lg:hidden"
            aria-label="Open navigation"
          >
            <Menu className="size-4" aria-hidden />
          </button>

          <button
            type="button"
            onClick={() => setCommandPaletteOpen(true)}
            className="flex h-8 flex-1 max-w-sm items-center gap-2 rounded-[var(--radius-md)] border border-border bg-surface-2 px-3 text-sm text-ink-muted transition-colors hover:border-border-strong"
          >
            <Search className="size-3.5" aria-hidden />
            <span className="flex-1 text-left">Search or ask AI...</span>
            <Kbd>⌘K</Kbd>
          </button>

          <div className="ml-auto flex items-center gap-1.5">
            <Tooltip content="Back to upload resume">
              <NavLink
                to={paths.onboarding}
                className="flex size-8 items-center justify-center rounded-[var(--radius-md)] text-ink-secondary transition-colors hover:bg-surface-2 hover:text-ink"
                aria-label="Back to upload resume"
              >
                <FileUp className="size-4" aria-hidden />
              </NavLink>
            </Tooltip>

            <Tooltip content="Notifications">
              <NavLink
                to={paths.notifications}
                className="relative flex size-8 items-center justify-center rounded-[var(--radius-md)] text-ink-secondary transition-colors hover:bg-surface-2 hover:text-ink"
              >
                <Bell className="size-4" aria-hidden />
                {unreadCount > 0 && (
                  <span className="absolute right-1 top-1 flex size-3.5 items-center justify-center rounded-full bg-danger text-[9px] font-semibold text-white">
                    {unreadCount > 9 ? "9+" : unreadCount}
                  </span>
                )}
              </NavLink>
            </Tooltip>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button type="button" className="ml-1 rounded-full outline-none focus-visible:ring-2 focus-visible:ring-brand-soft">
                  <Avatar name={user.email} size="sm" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                <div className="px-2.5 py-1.5 text-xs text-ink-muted">{user.email}</div>
                <DropdownMenuSeparator />
                <DropdownMenuItem onSelect={() => navigate(paths.settingsProfile)}>
                  <User className="size-3.5" aria-hidden /> Profile
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={() => navigate(paths.admin)}>
                  <Shield className="size-3.5" aria-hidden /> Admin portal
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem destructive onSelect={handleLogout}>
                  <LogOut className="size-3.5" aria-hidden /> Log out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </header>

        <main className="flex-1 px-5 py-6 sm:px-8">
          <div className="mx-auto w-full max-w-6xl">
            <PageTransition>
              <Suspense fallback={<PageLoader />}>
                <Outlet />
              </Suspense>
            </PageTransition>
          </div>
        </main>
      </div>

      <CommandPalette />
      <KeyboardShortcutsDialog open={shortcutsOpen} onOpenChange={setShortcutsOpen} />
    </div>
  );
}
