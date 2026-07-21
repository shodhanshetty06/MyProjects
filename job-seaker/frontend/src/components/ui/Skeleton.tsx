import { cn } from "@/lib/cn";

export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      role="presentation"
      className={cn(
        "rounded-[var(--radius-sm)] bg-[linear-gradient(90deg,var(--color-surface-2)_25%,var(--color-surface-hover)_37%,var(--color-surface-2)_63%)]",
        "bg-[length:400%_100%] animate-[shimmer_1.6s_linear_infinite]",
        className,
      )}
    />
  );
}

export function SkeletonText({ lines = 3, className }: { lines?: number; className?: string }) {
  return (
    <div className={cn("flex flex-col gap-2", className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton key={i} className={cn("h-3", i === lines - 1 ? "w-2/3" : "w-full")} />
      ))}
    </div>
  );
}

export function SkeletonCard({ className }: { className?: string }) {
  return (
    <div className={cn("rounded-[var(--radius-lg)] border border-border bg-surface p-5", className)}>
      <div className="flex items-center gap-3">
        <Skeleton className="size-9 rounded-full" />
        <div className="flex-1 space-y-2">
          <Skeleton className="h-3 w-1/3" />
          <Skeleton className="h-2.5 w-1/2" />
        </div>
      </div>
      <SkeletonText lines={2} className="mt-4" />
    </div>
  );
}
