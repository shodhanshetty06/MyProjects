import type { AnalyticsFiltersParams, TrendInterval } from "@/features/analytics/types";

export interface DateRangePreset {
  label: string;
  days: number | null;
}

const PRESETS: DateRangePreset[] = [
  { label: "Last 7 days", days: 7 },
  { label: "Last 30 days", days: 30 },
  { label: "Last 90 days", days: 90 },
  { label: "Last 12 months", days: 365 },
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
    <div className="analytics-filterbar">
      <div className="analytics-filterbar__group" role="group" aria-label="Date range">
        {PRESETS.map((preset) => (
          <button
            key={preset.label}
            type="button"
            className={`analytics-chip${activePresetDays === preset.days ? " analytics-chip--active" : ""}`}
            onClick={() => preset.days !== null && onPresetSelect(preset.days)}
          >
            {preset.label}
          </button>
        ))}
        <input
          type="date"
          aria-label="Custom start date"
          className="analytics-date-input"
          value={filters.date_from ? filters.date_from.slice(0, 10) : ""}
          max={filters.date_to ? filters.date_to.slice(0, 10) : toIsoDate(new Date())}
          onChange={(event) =>
            onCustomRangeChange(
              event.target.value ? `${event.target.value}T00:00:00Z` : undefined,
              filters.date_to,
            )
          }
        />
        <span className="analytics-filterbar__sep">to</span>
        <input
          type="date"
          aria-label="Custom end date"
          className="analytics-date-input"
          value={filters.date_to ? filters.date_to.slice(0, 10) : ""}
          max={toIsoDate(new Date())}
          onChange={(event) =>
            onCustomRangeChange(
              filters.date_from,
              event.target.value ? `${event.target.value}T23:59:59Z` : undefined,
            )
          }
        />
      </div>

      <select
        aria-label="Trend granularity"
        className="analytics-select"
        value={filters.interval ?? "week"}
        onChange={(event) => onIntervalChange(event.target.value as TrendInterval)}
      >
        <option value="day">Daily</option>
        <option value="week">Weekly</option>
        <option value="month">Monthly</option>
      </select>

      <input
        type="search"
        className="analytics-text-input"
        placeholder="Filter by company"
        aria-label="Filter by company"
        value={filters.company_name ?? ""}
        onChange={(event) => onCompanyNameChange(event.target.value)}
      />

      <div className="analytics-filterbar__spacer" />

      <div className="analytics-filterbar__exports">
        <button type="button" className="analytics-button" onClick={onExportCsv} disabled={isExportingCsv}>
          {isExportingCsv ? "Exporting…" : "Export CSV"}
        </button>
        <button type="button" className="analytics-button" onClick={onExportPdf} disabled={isExportingPdf}>
          {isExportingPdf ? "Exporting…" : "Export PDF"}
        </button>
      </div>
    </div>
  );
}
