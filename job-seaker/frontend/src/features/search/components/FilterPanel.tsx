import { useQuery } from "@tanstack/react-query";
import { X } from "lucide-react";
import { useEffect, useRef } from "react";

import { Checkbox } from "@/components/ui/Checkbox";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { useAuth } from "@/features/auth";
import {
  DATE_POSTED_OPTIONS,
  EMPLOYMENT_TYPES,
  EXPERIENCE_LEVELS,
  WORK_MODES,
  type DatePosted,
  type EmploymentType,
  type ExperienceLevel,
  type SearchFiltersParams,
  type WorkMode,
} from "@/features/search/types";
import { getPreferences } from "@/features/matching/api";
import { formatEnumLabel } from "@/lib/format";

interface FilterPanelProps {
  filters: SearchFiltersParams;
  onChange: (filters: SearchFiltersParams) => void;
}

/** Filter keys that count as "the user has touched filters" — if all of these are empty, it's safe to pre-populate from preferences. */
function hasActiveFilters(filters: SearchFiltersParams): boolean {
  return Boolean(
    filters.work_mode?.length ||
      filters.employment_type?.length ||
      filters.experience_level?.length ||
      filters.salary_min ||
      filters.country?.length,
  );
}

function toggleInArray<T>(arr: T[] | undefined, value: T): T[] {
  const list = arr ?? [];
  return list.includes(value) ? list.filter((v) => v !== value) : [...list, value];
}

const DATE_POSTED_LABELS: Record<DatePosted, string> = {
  any: "Any time",
  last_24h: "Past 24 hours",
  last_7d: "Past week",
  last_30d: "Past month",
};

/**
 * `MatchPreference` stores `years_of_experience` (a number), not an
 * `ExperienceLevel` enum value — this is a minimal, documented heuristic
 * bucketing so the filter panel can still pre-populate "Experience level"
 * from it, not a field that exists 1:1 on the backend model.
 */
function yearsToExperienceLevel(years: number | null): ExperienceLevel | null {
  if (years == null) return null;
  if (years < 1) return "internship";
  if (years < 2) return "entry";
  if (years < 5) return "mid";
  if (years < 8) return "senior";
  if (years < 12) return "lead";
  return "executive";
}

export function FilterPanel({ filters, onChange }: FilterPanelProps) {
  const { user } = useAuth();
  const appliedPrefillRef = useRef(false);

  const preferencesQuery = useQuery({
    queryKey: ["match-preferences", user?.id],
    queryFn: () => getPreferences(user!.id),
    enabled: !!user?.id,
    staleTime: 5 * 60_000,
    retry: false,
  });

  // Pre-populate from the user's match preferences on first mount, only if
  // filters are still empty/default — user edits after that always win,
  // this never overwrites a filter the user already touched.
  useEffect(() => {
    if (appliedPrefillRef.current) return;
    if (!preferencesQuery.data) return;
    if (hasActiveFilters(filters)) {
      appliedPrefillRef.current = true;
      return;
    }
    appliedPrefillRef.current = true;
    const prefs = preferencesQuery.data;
    const inferredLevel = yearsToExperienceLevel(prefs.years_of_experience);
    onChange({
      ...filters,
      work_mode: prefs.preferred_work_modes.length > 0 ? prefs.preferred_work_modes : filters.work_mode,
      employment_type: prefs.preferred_employment_types.length > 0 ? prefs.preferred_employment_types : filters.employment_type,
      country: prefs.preferred_countries.length > 0 ? prefs.preferred_countries : filters.country,
      salary_min: prefs.expected_salary_min ?? filters.salary_min,
      experience_level: inferredLevel ? [inferredLevel] : filters.experience_level,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preferencesQuery.data]);

  const activeCount =
    (filters.work_mode?.length ?? 0) +
    (filters.employment_type?.length ?? 0) +
    (filters.experience_level?.length ?? 0) +
    (filters.salary_min ? 1 : 0) +
    (filters.date_posted && filters.date_posted !== "any" ? 1 : 0);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-ink">Filters</h3>
        {activeCount > 0 && (
          <button
            type="button"
            onClick={() => onChange({ q: filters.q, mode: filters.mode, sort_by: filters.sort_by })}
            className="flex items-center gap-1 text-xs font-medium text-ink-muted hover:text-ink"
          >
            <X className="size-3" aria-hidden />
            Clear ({activeCount})
          </button>
        )}
      </div>

      <div className="space-y-2">
        <p className="text-xs font-medium text-ink-secondary">Work mode</p>
        {WORK_MODES.map((mode: WorkMode) => (
          <label key={mode} className="flex items-center gap-2 text-sm text-ink">
            <Checkbox
              checked={filters.work_mode?.includes(mode) ?? false}
              onCheckedChange={() => onChange({ ...filters, work_mode: toggleInArray(filters.work_mode, mode) })}
            />
            {formatEnumLabel(mode)}
          </label>
        ))}
      </div>

      <div className="space-y-2">
        <p className="text-xs font-medium text-ink-secondary">Employment type</p>
        {EMPLOYMENT_TYPES.map((type: EmploymentType) => (
          <label key={type} className="flex items-center gap-2 text-sm text-ink">
            <Checkbox
              checked={filters.employment_type?.includes(type) ?? false}
              onCheckedChange={() =>
                onChange({ ...filters, employment_type: toggleInArray(filters.employment_type, type) })
              }
            />
            {formatEnumLabel(type)}
          </label>
        ))}
      </div>

      <div className="space-y-2">
        <p className="text-xs font-medium text-ink-secondary">Experience level</p>
        {EXPERIENCE_LEVELS.map((level: ExperienceLevel) => (
          <label key={level} className="flex items-center gap-2 text-sm text-ink">
            <Checkbox
              checked={filters.experience_level?.includes(level) ?? false}
              onCheckedChange={() =>
                onChange({ ...filters, experience_level: toggleInArray(filters.experience_level, level) })
              }
            />
            {formatEnumLabel(level)}
          </label>
        ))}
      </div>

      <div className="space-y-2">
        <p className="text-xs font-medium text-ink-secondary">Salary (min)</p>
        <Input
          type="number"
          placeholder="e.g. 80000"
          value={filters.salary_min ?? ""}
          onChange={(e) => onChange({ ...filters, salary_min: e.target.value ? Number(e.target.value) : undefined })}
        />
      </div>

      <div className="space-y-2">
        <p className="text-xs font-medium text-ink-secondary">Date posted</p>
        <Select
          value={filters.date_posted ?? "any"}
          onValueChange={(v) => onChange({ ...filters, date_posted: v as DatePosted })}
          options={DATE_POSTED_OPTIONS.map((d) => ({ value: d, label: DATE_POSTED_LABELS[d] }))}
        />
      </div>
    </div>
  );
}
