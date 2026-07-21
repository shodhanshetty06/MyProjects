import { useQuery } from "@tanstack/react-query";
import { Building2, Search, Sparkles, Tag } from "lucide-react";
import { useRef, useState } from "react";

import { getAutocomplete } from "@/features/search/api";
import type { SearchMode } from "@/features/search/types";
import { cn } from "@/lib/cn";
import { useDebouncedValue } from "@/lib/useDebouncedValue";

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (value: string) => void;
  mode: SearchMode;
  onModeChange: (mode: SearchMode) => void;
}

const MODES: { value: SearchMode; label: string; hint: string }[] = [
  { value: "keyword", label: "Keyword", hint: "Exact term matching" },
  { value: "semantic", label: "Semantic", hint: "AI meaning-based match" },
  { value: "hybrid", label: "Hybrid", hint: "Best of both" },
];

const SUGGESTION_ICON = { job_title: Search, company: Building2, skill: Tag };

export function SearchBar({ value, onChange, onSubmit, mode, onModeChange }: SearchBarProps) {
  const [focused, setFocused] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const debouncedValue = useDebouncedValue(value, 250);

  const suggestionsQuery = useQuery({
    queryKey: ["autocomplete", debouncedValue],
    queryFn: () => getAutocomplete(debouncedValue, 8),
    enabled: debouncedValue.trim().length >= 2 && focused,
    staleTime: 60_000,
  });

  const suggestions = suggestionsQuery.data ?? [];

  return (
    <div className="relative">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-ink-muted" aria-hidden />
          <input
            ref={inputRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onFocus={() => setFocused(true)}
            onBlur={() => window.setTimeout(() => setFocused(false), 120)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                onSubmit(value);
                setFocused(false);
              }
            }}
            placeholder="Search job titles, companies, or skills..."
            className="h-12 w-full rounded-[var(--radius-lg)] border border-border bg-surface pl-10 pr-4 text-sm text-ink placeholder:text-ink-muted outline-none transition-[border-color,box-shadow] focus:border-brand focus:ring-4 focus:ring-brand-soft"
          />
        </div>

        <div className="inline-flex items-center gap-1 rounded-[var(--radius-lg)] border border-border bg-surface-2 p-1">
          {MODES.map((m) => (
            <button
              key={m.value}
              type="button"
              onClick={() => onModeChange(m.value)}
              title={m.hint}
              className={cn(
                "flex items-center gap-1 rounded-[var(--radius-md)] px-3 py-2 text-xs font-medium transition-colors",
                mode === m.value ? "bg-surface text-ink shadow-xs" : "text-ink-secondary hover:text-ink",
              )}
            >
              {m.value === "semantic" && <Sparkles className="size-3" aria-hidden />}
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {focused && value.trim().length >= 2 && suggestions.length > 0 && (
        <div className="absolute left-0 right-0 top-[calc(100%+6px)] z-20 overflow-hidden rounded-[var(--radius-lg)] border border-border bg-surface shadow-lg">
          {suggestions.map((s, i) => {
            const Icon = SUGGESTION_ICON[s.type];
            return (
              <button
                key={`${s.type}-${s.label}-${i}`}
                type="button"
                onMouseDown={(e) => {
                  e.preventDefault();
                  onChange(s.label);
                  onSubmit(s.label);
                  setFocused(false);
                }}
                className="flex w-full items-center gap-2.5 px-4 py-2.5 text-left text-sm text-ink transition-colors hover:bg-surface-2"
              >
                <Icon className="size-3.5 text-ink-muted" aria-hidden />
                {s.label}
                <span className="ml-auto text-[10px] uppercase tracking-wide text-ink-muted">{s.type.replace("_", " ")}</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
