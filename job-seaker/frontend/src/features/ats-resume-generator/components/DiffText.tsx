import { cn } from "@/lib/cn";
import type { TextDiffSegment } from "@/features/ats-resume-generator/types";

export function DiffText({ segments }: { segments: TextDiffSegment[] }) {
  if (segments.length === 0) return <span className="text-ink-muted">No content</span>;
  return (
    <>
      {segments.map((segment, i) => (
        <span
          key={i}
          className={cn(
            segment.op === "insert" && "bg-success-soft text-success",
            segment.op === "delete" && "bg-danger-soft text-danger line-through",
            segment.op === "equal" && "text-ink-secondary",
          )}
        >
          {segment.text}
        </span>
      ))}
    </>
  );
}
