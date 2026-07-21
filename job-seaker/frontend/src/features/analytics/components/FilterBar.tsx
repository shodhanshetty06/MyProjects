import { Download, Search } from "lucide-react";

import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import type { AnalyticsFiltersParams, TrendInterval } from "@/features/analytics/types";
import { cn } from "@/lib/cn";

export interface DateRangePreset {
  label: string;
  days: number | null;
}

const PRESETS: DateRangePreset[] = [
  { label: "7d", days: 7 },
  { label: "30d", days: 30 },
  { label: "90d", days: 90 },
  { label: "12mo", days: 365 },
];

const INTERVAL_OPTIONS = [
  { value: "day", label: "Daily" },
  { value: "week", label: "Weekly" },
  { value: "month", label: "Monthly" },
];

function toIsoDate(date: Date): string {
  return date.toISOString().slice(0, 10);
}

interface FilterBarProps {
  filters: AnalyticsFiltersParams;
  activePresetDays: number | null;
  onPresetSelect: (days: number) => void;
  onCustomRangeChange: (dateFrom: string | undefined, dateTo: string | undefined) => void;
  onIntervalChange: (interval: TrendInterval) => void;
  onCompanyNameChange: (companyName: string) => void;
  onExportCsv: () => void;
  onExportPdf: () => void;
  isExportingCsv: boolean;
  isExportingPdf: boolean;
}

export function FilterBar({
  filters,
  activePresetDays,
  onPresetSelect,
  onCustomRangeChange,
  onIntervalChange,
  onCompanyNameChange,
  onExportCsv,
  onExportPdf,
  isExportingCsv,
  isExportingPdf,
}: FilterBarProps) {
  return (
    <div className="mb-6 flex flex-wrap items-center gap-2 rounded-[var(--radius-lg)] border border-border bg-surface p-3">
      <div className="flex items-center gap-1 rounded-[var(--radius-md)] bg-surface-2 p-1" role="group" aria-label="Date range">
        {PRESETS.map((preset) => (
          <button
            key={preset.label}
            type="button"
            onClick={() => preset.days !== null && onPresetSelect(preset.days)}
            className={cn(
              "rounded-[var(--radius-sm)] px-2.5 py-1.5 text-xs font-medium transition-colors",
              activePresetDays === preset.days ? "bg-surface text-ink shadow-xs" : "text-ink-secondary hover:text-ink",
            )}
          >
            {preset.label}
          </button>
        ))}
      </div>

      <input
        type="date"
        aria-label="Custom start date"
        className="h-8 rounded-[var(--radius-md)] border border-border bg-surface px-2 text-xs text-ink outline-none focus:border-brand"
        value={filters.date_from ? filters.date_from.slice(0, 10) : ""}
        max={filters.date_to ? filters.date_to.slice(0, 10) : toIsoDate(new Date())}
        onChange={(event) =>
          onCustomRangeChange(event.target.value ? `${event.target.value}T00:00:00Z` : undefined, filters.date_to)
        }
      />
      <span className="text-xs text-ink-muted">to</span>
      <input
        type="date"
        aria-label="Custom end date"
        className="h-8 rounded-[var(--radius-md)] border border-border bg-surface px-2 text-xs text-ink outline-none focus:border-brand"
        value={filters.date_to ? filters.date_to.slice(0, 10) : ""}
        max={toIsoDate(new Date())}
        onChange={(event) =>
          onCustomRangeChange(filters.date_from, event.target.value ? `${event.target.value}T23:59:59Z` : undefined)
        }
      />

      <Select
        value={filters.interval ?? "week"}
        onValueChange={(v) => onIntervalChange(v as TrendInterval)}
        options={INTERVAL_OPTIONS}
        className="w-32"
      />

      <Input
        leadingIcon={<Search />}
        placeholder="Filter by company"
        value={filters.company_name ?? ""}
        onChange={(event) => onCompanyNameChange(event.target.value)}
        className="w-44"
      />

      <div className="ml-auto flex items-center gap-2">
        <Button variant="secondary" size="sm" onClick={onExportCsv} isLoading={isExportingCsv}>
          <Download className="size-3.5" aria-hidden />
          CSV
        </Button>
        <Button variant="secondary" size="sm" onClick={onExportPdf} isLoading={isExportingPdf}>
          <Download className="size-3.5" aria-hidden />
          PDF
        </Button>
      </div>
    </div>
  );
}
