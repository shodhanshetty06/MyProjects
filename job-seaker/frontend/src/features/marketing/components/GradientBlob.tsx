import { cn } from "@/lib/cn";

export function GradientBlob({ className }: { className?: string }) {
  return (
    <div
      aria-hidden
      className={cn(
        "pointer-events-none absolute rounded-full opacity-20 blur-3xl",
        className,
      )}
      style={{ background: "linear-gradient(135deg,var(--color-accent-a),var(--color-accent-b))" }}
    />
  );
}
