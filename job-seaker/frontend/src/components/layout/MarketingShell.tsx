import { Menu, Sparkles, X } from "lucide-react";
import { Suspense, useState } from "react";
import { Link, NavLink, Outlet } from "react-router-dom";

import { PageLoader } from "@/components/common/LoadingSpinner";
import { Button } from "@/components/ui/Button";
import { useAuth } from "@/features/auth";
import { cn } from "@/lib/cn";
import { paths } from "@/routes/paths";

const NAV_LINKS = [
  { to: paths.features, label: "Features" },
  { to: paths.pricing, label: "Pricing" },
  { to: paths.about, label: "About" },
  { to: paths.contact, label: "Contact" },
];

export function MarketingShell() {
  const { isAuthenticated } = useAuth();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="flex min-h-screen flex-col bg-bg">
      <header className="sticky top-0 z-40 border-b border-border bg-surface/80 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-5 sm:px-8">
          <Link to={paths.home} className="flex items-center gap-2">
            <div className="flex size-7 items-center justify-center rounded-[var(--radius-sm)] bg-[linear-gradient(135deg,var(--color-accent-a),var(--color-accent-b))]">
              <Sparkles className="size-4 text-white" aria-hidden />
            </div>
            <span className="text-sm font-semibold text-ink">Ascend AI</span>
          </Link>

          <nav className="hidden items-center gap-1 md:flex" aria-label="Marketing navigation">
            {NAV_LINKS.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                className={({ isActive }) =>
                  cn(
                    "rounded-[var(--radius-md)] px-3 py-1.5 text-sm font-medium transition-colors",
                    isActive ? "text-ink" : "text-ink-secondary hover:text-ink",
                  )
                }
              >
                {link.label}
              </NavLink>
            ))}
          </nav>

          <div className="hidden items-center gap-2 md:flex">
            {isAuthenticated ? (
              <Link to={paths.dashboard}>
                <Button size="sm">Go to dashboard</Button>
              </Link>
            ) : (
              <>
                <Link to={paths.login}>
                  <Button variant="ghost" size="sm">
                    Log in
                  </Button>
                </Link>
                <Link to={paths.register}>
                  <Button size="sm">Get started</Button>
                </Link>
              </>
            )}
          </div>

          <button
            type="button"
            className="flex size-9 items-center justify-center rounded-[var(--radius-md)] text-ink md:hidden"
            onClick={() => setMobileOpen((v) => !v)}
            aria-label="Toggle menu"
          >
            {mobileOpen ? <X className="size-5" /> : <Menu className="size-5" />}
          </button>
        </div>

        {mobileOpen && (
          <div className="flex flex-col gap-1 border-t border-border px-5 py-3 md:hidden">
            {NAV_LINKS.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                onClick={() => setMobileOpen(false)}
                className="rounded-[var(--radius-md)] px-3 py-2 text-sm font-medium text-ink-secondary hover:bg-surface-2 hover:text-ink"
              >
                {link.label}
              </NavLink>
            ))}
            <div className="mt-2 flex flex-col gap-2 border-t border-border pt-3">
              <Link to={paths.login} onClick={() => setMobileOpen(false)}>
                <Button variant="secondary" className="w-full">
                  Log in
                </Button>
              </Link>
              <Link to={paths.register} onClick={() => setMobileOpen(false)}>
                <Button className="w-full">Get started</Button>
              </Link>
            </div>
          </div>
        )}
      </header>

      <main className="flex-1">
        <Suspense fallback={<PageLoader />}>
          <Outlet />
        </Suspense>
      </main>

      <footer className="border-t border-border">
        <div className="mx-auto flex max-w-6xl flex-col gap-6 px-5 py-10 sm:px-8 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-2">
            <div className="flex size-6 items-center justify-center rounded-[var(--radius-sm)] bg-[linear-gradient(135deg,var(--color-accent-a),var(--color-accent-b))]">
              <Sparkles className="size-3.5 text-white" aria-hidden />
            </div>
            <span className="text-sm font-medium text-ink">Ascend AI</span>
            <span className="text-sm text-ink-muted">&copy; {new Date().getFullYear()}</span>
          </div>
          <nav className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-ink-secondary">
            {NAV_LINKS.map((link) => (
              <Link key={link.to} to={link.to} className="hover:text-ink">
                {link.label}
              </Link>
            ))}
          </nav>
        </div>
      </footer>
    </div>
  );
}
