import { Check, Trash2 } from "lucide-react";
import type { ReactNode } from "react";

import { Button } from "@/components/ui/Button";
import { ReviewBadge, type ReviewFieldState } from "@/features/onboarding/components/ReviewBadge";

interface ReviewItemCardProps {
  state: Exclude<ReviewFieldState, "missing">;
  onAccept?: () => void;
  onDelete: () => void;
  children: ReactNode;
}

/**
 * One editable entry inside a repeatable section (an experience, education,
 * project, certification, or language item). Every field inside `children`
 * is a plain editable input — this card only supplies the surrounding
 * Accept/Delete actions and the AI/Accepted/Added badge; "Edit" is just
 * typing into those inputs, and "Add" is the section's own button that
 * appends a new card in the "added" state.
 */
export function ReviewItemCard({ state, onAccept, onDelete, children }: ReviewItemCardProps) {
  return (
    <div className="space-y-3 rounded-[var(--radius-lg)] border border-border bg-surface p-4">
      <div className="flex items-center justify-between gap-2">
        <ReviewBadge state={state} />
        <div className="flex items-center gap-1.5">
          {state === "ai" && onAccept && (
            <Button variant="secondary" size="sm" onClick={onAccept} type="button">
              <Check className="size-3.5" aria-hidden />
              Accept
            </Button>
          )}
          <Button variant="ghost" size="sm" onClick={onDelete} type="button" aria-label="Delete entry">
            <Trash2 className="size-3.5" aria-hidden />
            Delete
          </Button>
        </div>
      </div>
      {children}
    </div>
  );
}
