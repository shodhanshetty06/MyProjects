import { useMutation, useQuery } from "@tanstack/react-query";
import { Building2, Loader2, Search, Sparkles, Tag, X } from "lucide-react";
import { useRef, useState } from "react";

import { getAutocomplete, parseSearchQuery } from "@/features/search/api";
import type { ParsedSearchQuery, SearchMode } from "@/features/search/types";
import { cn } from "@/lib/cn";
import { useDebouncedValue } from "@/lib/useDebouncedValue";

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (cleanedQuery: string, parsed?: ParsedSearchQuery) => void;
  mode: SearchMode;
  onModeChange: (mode: SearchMode) => void;
}

const MODES: { value: SearchMode; label: string; hint: string }[] = [
  { value: "keyword", label: "Keyword", hint: "Exact term matching" },
  { value: "semantic", label: "Semantic", hint: "AI meaning-based match" },
  { value: "hybrid", label: "Hybrid", hint: "Best of both" },
];

const SUGGESTION_ICON = { job_title: Search, company: Building2, skill: Tag };

/** Turns a `ParsedSearchQuery` into a flat list of "chip" labels the user
 * can see (and, via `onRemove`, undo) before/after search runs — the
 * transparency step described in `nl_query.py`'s module docstring: a
 * misparsed filter should never silently narrow results without the user
 * knowing why. */
function parsedQueryChips(parsed: ParsedSearchQuery): { key: string; label: string }[] {
  const chips: { key: string; label: string }[] = [];
  if (parsed.work_mode) chips.push({ key: "work_mode", label: parsed.work_mode });
  for (const skill of parsed.skills) chips.push({ key: `skill:${skill}`, label: skill });
  if (parsed.salary_min != null) {
    chips.push({
      key: "salary_min",
      label: `${parsed.salary_currency ?? ""} ${parsed.salary_min.toLocaleString()}+`.trim(),
    });
  }
  if (parsed.experience_level) chips.push({ key: "experience_level", label: parsed.experience_level });
  if (parsed.employment_type) chips.push({ key: "employment_type", label: parsed.employment_type.replace("_", " ") });
  if (parsed.city) chips.push({ key: "city", label: parsed.city });
  if (parsed.country) chips.push({ key: "country", label: parsed.country });
  return chips;
}

export function SearchBar({ value, onChange, onSubmit, mode, onModeChange }: SearchBarProps) {
  const [focused, setFocused] = useState(false);
  const [parsed, setParsed] = useState<ParsedSearchQuery | null>(null);
  const [removedChipKeys, setRemovedChipKeys] = useState<Set<string>>(new Set());
  const inputRef = useRef<HTMLInputElement>(null);
  const debouncedValue = useDebouncedValue(value, 250);

  const suggestionsQuery = useQuery({
    queryKey: ["autocomplete", debouncedValue],
    queryFn: () => getAutocomplete(debouncedValue, 8),
    enabled: debouncedValue.trim().length >= 2 && focused,
    staleTime: 60_000,
  });

  const parseMutation = useMutation({
    mutationFn: (q: string) => parseSearchQuery(q),
  });

  const suggestions = suggestionsQuery.data ?? [];

  function applyParsed(raw: string, result: ParsedSearchQuery) {
    setParsed(result);
    setRemovedChipKeys(new Set());
    onSubmit(result.cleaned_query || raw, result);
  }

  async function submit(raw: string) {
    setFocused(false);
    if (!raw.trim()) {
      setParsed(null);
      onSubmit(raw);
      return;
    }
    try {
      const result = await parseMutation.mutateAsync(raw);
      applyParsed(raw, result);
    } catch {
      // Server-side parsing failed (AI provider hiccup, network) — degrade
      // to plain keyword/semantic search on the untouched query, same
      // behavior as before this feature existed.
      setParsed(null);
      onSubmit(raw);
    }
  }

  function removeChip(key: string) {
    if (!parsed) return;
    const next = new Set(removedChipKeys);
    next.add(key);
    setRemovedChipKeys(next);
    // Re-issue the search with that one filter cleared, keeping the rest.
    const stripped: ParsedSearchQuery = { ...parsed };
    if (key === "work_mode") stripped.work_mode = null;
    else if (key === "salary_min") {
      stripped.salary_min = null;
      stripped.salary_currency = null;
    } else if (key === "experience_level") stripped.experience_level = null;
    else if (key === "employment_type") stripped.employment_type = null;
    else if (key === "city") stripped.city = null;
    else if (key === "country") stripped.country = null;
    else if (key.startsWith("skill:")) {
      const skill = key.slice("skill:".length);
      stripped.skills = stripped.skills.filter((s) => s !== skill);
    }
    setParsed(stripped);
    onSubmit(stripped.cleaned_query || value, stripped);
  }

  const chips = parsed ? parsedQueryChips(parsed).filter((c) => !removedChipKeys.has(c.key)) : [];

  return (
    <div className="relative">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <div className="relative flex-1">
          {parseMutation.isPending ? (
            <Loader2 className="pointer-events-none absolute left-3.5 top-1/2 size-4 -translate-y-1/2 animate-spin text-ink-muted" aria-hidden />
          ) : (
            <Search className="pointer-events-none absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-ink-muted" aria-hidden />
          )}
          <input
            ref={inputRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onFocus={() => setFocused(true)}
            onBlur={() => window.setTimeout(() => setFocused(false), 120)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                void submit(value);
              }
            }}
            placeholder="Try: Find remote Python jobs paying above 20 LPA, Senior FastAPI roles, AI startups hiring..."
            className="h-12 w-full rounded-[var(--radius-lg)] border border-border bg-surface pl-10 pr-4 text-sm text-ink placeholder:text-ink-muted outline-none transition-[border-color,box-shadow] focus:border-brand focus:ring-4 focus:ring-brand-soft"
          />
        </div>

        <button
          type="button"
          onClick={() => void submit(value)}
          className="h-12 shrink-0 rounded-[var(--radius-lg)] bg-brand px-5 text-sm font-medium text-white transition-colors hover:bg-brand/90"
        >
          Search
        </button>

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

      {chips.length > 0 && (
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          <span className="flex items-center gap-1 text-xs text-ink-muted">
            <Sparkles className="size-3" aria-hidden />
            Understood as:
          </span>
          {chips.map((chip) => (
            <span
              key={chip.key}
              className="inline-flex items-center gap-1 rounded-full bg-brand-soft px-2.5 py-0.5 text-xs font-medium text-brand"
            >
              {chip.label}
              <button
                type="button"
                onClick={() => removeChip(chip.key)}
                aria-label={`Remove ${chip.label} filter`}
                className="rounded-full hover:bg-brand/20"
              >
                <X className="size-3" aria-hidden />
              </button>
            </span>
          ))}
        </div>
      )}

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
                  void submit(s.label);
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
