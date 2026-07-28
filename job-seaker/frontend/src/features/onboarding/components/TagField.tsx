import { Plus, X } from "lucide-react";
import { useState, type KeyboardEvent } from "react";

import { Input } from "@/components/ui/Input";
import { cn } from "@/lib/cn";

interface TagFieldProps {
  values: string[];
  onChange: (values: string[]) => void;
  placeholder?: string;
  disabled?: boolean;
}

/**
 * Chip editor for string-list fields (skills, achievements, target roles).
 * Enter/comma commits the current input as a new chip; each chip has its
 * own delete button — this is the "Edit"/"Delete"/"Add" surface for
 * list-of-strings sections of the Profile Review screen.
 */
export function TagField({ values, onChange, placeholder, disabled }: TagFieldProps) {
  const [draft, setDraft] = useState("");

  function commit() {
    const value = draft.trim();
    if (value && !values.includes(value)) {
      onChange([...values, value]);
    }
    setDraft("");
  }

  function handleKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      commit();
    } else if (e.key === "Backspace" && !draft && values.length > 0) {
      onChange(values.slice(0, -1));
    }
  }

  return (
    <div
      className={cn(
        "flex flex-wrap items-center gap-1.5 rounded-[var(--radius-md)] border border-border bg-surface p-2",
        disabled && "opacity-50",
      )}
    >
      {values.map((value, index) => (
        <span
          key={`${value}-${index}`}
          className="inline-flex items-center gap-1 rounded-full bg-surface-2 px-2.5 py-1 text-xs font-medium text-ink"
        >
          {value}
          {!disabled && (
            <button
              type="button"
              onClick={() => onChange(values.filter((_, i) => i !== index))}
              className="text-ink-muted hover:text-danger"
              aria-label={`Remove ${value}`}
            >
              <X className="size-3" aria-hidden />
            </button>
          )}
        </span>
      ))}
      {!disabled && (
        <div className="flex min-w-32 flex-1 items-center gap-1">
          <Input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={handleKeyDown}
            onBlur={commit}
            placeholder={values.length === 0 ? placeholder : "Add another..."}
            className="h-7 border-none px-1 shadow-none hover:border-none focus:ring-0"
          />
          {draft.trim() && (
            <button
              type="button"
              onClick={commit}
              className="flex size-6 shrink-0 items-center justify-center rounded-full bg-brand-soft text-brand"
              aria-label="Add"
            >
              <Plus className="size-3.5" aria-hidden />
            </button>
          )}
        </div>
      )}
    </div>
  );
}
