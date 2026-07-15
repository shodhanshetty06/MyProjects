import { formatDateLabel } from "@/features/analytics/format";
import { chrome, pick, sequentialBlue, useColorScheme } from "@/features/analytics/theme";
import type { HeatmapCell } from "@/features/analytics/types";

interface HeatmapCalendarProps {
  cells: HeatmapCell[];
}

const LEVEL_STEPS = [sequentialBlue[200], sequentialBlue[400], sequentialBlue[500], sequentialBlue[700]];

function levelColor(count: number, max: number, emptyColor: string): string {
  if (count === 0 || max === 0) {
    return emptyColor;
  }
  const ratio = count / max;
  const index = Math.min(LEVEL_STEPS.length - 1, Math.floor(ratio * LEVEL_STEPS.length));
  return LEVEL_STEPS[index];
}

export function HeatmapCalendar({ cells }: HeatmapCalendarProps) {
  const scheme = useColorScheme();
  const emptyColor = pick(chrome.gridline, scheme);
  const muted = pick(chrome.mutedInk, scheme);

  if (cells.length === 0) {
    return <p className="analytics-empty">No data for this period.</p>;
  }

  const max = Math.max(...cells.map((c) => c.count));
  const firstWeekday = new Date(cells[0].date).getUTCDay();
  const padded: Array<HeatmapCell | null> = [...Array(firstWeekday).fill(null), ...cells];

  const weeks: Array<Array<HeatmapCell | null>> = [];
  for (let i = 0; i < padded.length; i += 7) {
    weeks.push(padded.slice(i, i + 7));
  }

  return (
    <div className="analytics-heatmap">
      <div className="analytics-heatmap__grid">
        {weeks.map((week, weekIndex) => (
          <div className="analytics-heatmap__week" key={weekIndex}>
            {week.map((cell, dayIndex) =>
              cell ? (
                <div
                  key={cell.date}
                  className="analytics-heatmap__cell"
                  style={{ backgroundColor: levelColor(cell.count, max, emptyColor) }}
                  title={`${formatDateLabel(cell.date)}: ${cell.count} application${cell.count === 1 ? "" : "s"}`}
                />
              ) : (
                <div key={dayIndex} className="analytics-heatmap__cell analytics-heatmap__cell--empty" />
              ),
            )}
          </div>
        ))}
      </div>
      <div className="analytics-heatmap__legend">
        <span style={{ color: muted }}>Less</span>
        <span className="analytics-heatmap__cell" style={{ backgroundColor: emptyColor }} />
        {LEVEL_STEPS.map((step) => (
          <span className="analytics-heatmap__cell" style={{ backgroundColor: step }} key={step} />
        ))}
        <span style={{ color: muted }}>More</span>
      </div>
    </div>
  );
}
