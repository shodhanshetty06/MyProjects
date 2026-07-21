import { cn } from "@/lib/cn";

export function Kbd({ children, className }: { children: string; className?: string }) {
  return (
    <kbd
      className={cn(
        "inline-flex h-5 min-w-5 items-center justify-center rounded-[4px] border border-border-strong bg-surface-2 px-1 font-mono text-[10px] font-medium text-ink-secondary",
        className,
      )}
    >
      {children}
    </kbd>
  );
}
