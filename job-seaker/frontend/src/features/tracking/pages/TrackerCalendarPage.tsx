import { useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { SkeletonText } from "@/components/ui/Skeleton";
import { useAuth } from "@/features/auth";
import { listApplications } from "@/features/tracking/api";
import { ApplicationStatusBadge } from "@/features/tracking/components/ApplicationStatusBadge";
import type { ApplicationRead } from "@/features/tracking/types";
import { cn } from "@/lib/cn";
import { paths } from "@/routes/paths";

const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

export function TrackerCalendarPage() {
  const { user } = useAuth();
  const userId = user!.id;
  const [cursor, setCursor] = useState(() => new Date());
  const [selectedDay, setSelectedDay] = useState<string | null>(null);

  const year = cursor.getFullYear();
  const month = cursor.getMonth();
  const monthStart = new Date(year, month, 1);
  const monthEnd = new Date(year, month + 1, 0);

  const applicationsQuery = useQuery({
    queryKey: ["applications", userId, "calendar", year, month],
    queryFn: () =>
      listApplications(userId, {
        applied_after: monthStart.toISOString(),
        applied_before: new Date(year, month + 1, 1).toISOString(),
        page_size: 100,
      }),
  });

  const byDay = useMemo(() => {
    const map = new Map<string, ApplicationRead[]>();
    for (const app of applicationsQuery.data?.data ?? []) {
      const key = app.applied_at.slice(0, 10);
      const list = map.get(key) ?? [];
      list.push(app);
      map.set(key, list);
    }
    return map;
  }, [applicationsQuery.data]);

  const leadingBlanks = monthStart.getDay();
  const daysInMonth = monthEnd.getDate();
  const cells: (number | null)[] = [
    ...Array.from({ length: leadingBlanks }, () => null),
    ...Array.from({ length: daysInMonth }, (_, i) => i + 1),
  ];

  const selectedApps = selectedDay ? (byDay.get(selectedDay) ?? []) : [];

  return (
    <div>
      <PageHeader
        eyebrow="Applications"
        title="Calendar"
        description="Application activity by day."
        actions={
          <div className="flex items-center gap-1">
            <Button variant="outline" size="sm" onClick={() => setCursor(new Date(year, month - 1, 1))}>
              <ChevronLeft className="size-4" aria-hidden />
            </Button>
            <span className="min-w-32 text-center text-sm font-medium text-ink">
              {monthStart.toLocaleDateString(undefined, { month: "long", year: "numeric" })}
            </span>
            <Button variant="outline" size="sm" onClick={() => setCursor(new Date(year, month + 1, 1))}>
              <ChevronRight className="size-4" aria-hidden />
            </Button>
          </div>
        }
      />

      {applicationsQuery.isLoading ? (
        <SkeletonText lines={6} />
      ) : (
        <div className="grid gap-6 lg:grid-cols-[1fr_280px]">
          <div>
            <div className="mb-1 grid grid-cols-7 gap-1.5 text-center text-xs font-medium text-ink-muted">
              {WEEKDAYS.map((d) => (
                <span key={d}>{d}</span>
              ))}
            </div>
            <div className="grid grid-cols-7 gap-1.5">
              {cells.map((day, i) => {
                if (!day) return <div key={i} />;
                const key = `${year}-${String(month + 1).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
                const count = byDay.get(key)?.length ?? 0;
                const isSelected = key === selectedDay;
                return (
                  <button
                    key={i}
                    onClick={() => setSelectedDay(count > 0 ? key : null)}
                    className={cn(
                      "flex h-16 flex-col items-start gap-1 rounded-[var(--radius-md)] border p-1.5 text-left transition-colors",
                      isSelected ? "border-brand-border bg-brand-soft" : "border-border bg-surface hover:bg-surface-2",
                      count === 0 && "cursor-default",
                    )}
                  >
                    <span className="text-xs text-ink-muted">{day}</span>
                    {count > 0 && (
                      <span className="rounded-full bg-brand px-1.5 py-0.5 text-[10px] font-medium text-brand-ink">
                        {count}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          <Card className="h-fit p-4">
            <h3 className="mb-3 text-sm font-semibold text-ink">
              {selectedDay ? new Date(selectedDay).toLocaleDateString(undefined, { weekday: "long", month: "short", day: "numeric" }) : "Select a day"}
            </h3>
            {selectedApps.length === 0 ? (
              <p className="text-xs text-ink-muted">No applications on this day.</p>
            ) : (
              <ul className="space-y-2">
                {selectedApps.map((app) => (
                  <li key={app.id}>
                    <Link to={paths.trackerTimeline(app.id)} className="block rounded-[var(--radius-sm)] p-1.5 hover:bg-surface-2">
                      <p className="truncate text-xs font-medium text-ink">{app.job_title}</p>
                      <p className="truncate text-[11px] text-ink-muted">{app.company_name}</p>
                      <ApplicationStatusBadge status={app.status} />
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </div>
      )}
    </div>
  );
}
