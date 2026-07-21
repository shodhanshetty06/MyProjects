import { X } from "lucide-react";

import { Checkbox } from "@/components/ui/Checkbox";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
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
import { formatEnumLabel } from "@/lib/format";

interface FilterPanelProps {
  filters: SearchFiltersParams;
  onChange: (filters: SearchFiltersParams) => void;
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

export function FilterPanel({ filters, onChange }: FilterPanelProps) {
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
