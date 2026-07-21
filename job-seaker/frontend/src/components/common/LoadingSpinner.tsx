import { Loader2 } from "lucide-react";

import { cn } from "@/lib/cn";

export function LoadingSpinner({ className }: { className?: string }) {
  return (
    <div role="status" aria-live="polite" className="inline-flex items-center gap-2 text-ink-secondary">
      <Loader2 className={cn("size-4 animate-spin", className)} aria-hidden />
      <span className="text-sm">Loading…</span>
    </div>
  );
}

/** Full-viewport fallback for route-level React.lazy() Suspense boundaries. */
export function PageLoader() {
  return (
    <div role="status" aria-live="polite" className="flex min-h-[60vh] w-full items-center justify-center">
      <Loader2 className="size-6 animate-spin text-brand" aria-hidden />
      <span className="sr-only">Loading page…</span>
    </div>
  );
}
