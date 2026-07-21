import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { getPlatformAnalytics } from "@/features/admin/api";
import { StatTile } from "@/features/admin/components/StatTile";
import { TrendChart } from "@/features/admin/components/TrendChart";

const DAY_OPTIONS = [7, 30, 90];

export function PlatformAnalyticsPage() {
  const [days, setDays] = useState(30);

  const { data, isLoading, isError } = useQuery({
    queryKey: ["admin", "analytics", days],
    queryFn: () => getPlatformAnalytics(days),
    placeholderData: keepPreviousData,
  });

  return (
    <section>
      <header className="admin-page__header">
        <h1>Analytics</h1>
        <p className="admin-page__subtitle">Platform-wide activity and moderation throughput.</p>
      </header>

      <div className="admin-toolbar">
        {DAY_OPTIONS.map((option) => (
          <button
            key={option}
            type="button"
            className={`admin-button${days === option ? "" : " admin-button--ghost"}`}
            onClick={() => setDays(option)}
          >
            Last {option} days
          </button>
        ))}
      </div>

      {isLoading && <p className="admin-status">Loading…</p>}
      {isError && <p className="admin-status admin-status--error">Unable to load analytics.</p>}

      {data && (
        <>
          <div className="admin-stat-grid">
            <StatTile label="Reports resolved" value={data.reports_resolved_count} />
            <StatTile label="Jobs moderated" value={data.jobs_moderated_count} />
            <StatTile label="Companies reviewed" value={data.companies_reviewed_count} />
          </div>

          <div className="admin-card admin-section">
            <h2 className="admin-section__title">Active users</h2>
            <TrendChart data={data.active_users_trend} color="var(--adm-blue)" />
          </div>
          <div className="admin-card admin-section">
            <h2 className="admin-section__title">Applications submitted</h2>
            <TrendChart data={data.applications_trend} color="var(--adm-green)" />
          </div>
          <div className="admin-card admin-section">
            <h2 className="admin-section__title">Jobs posted</h2>
            <TrendChart data={data.jobs_posted_trend} color="var(--adm-amber)" />
          </div>
          <div className="admin-card admin-section">
            <h2 className="admin-section__title">Companies added</h2>
            <TrendChart data={data.companies_added_trend} color="var(--color-accent-b)" />
          </div>
          <div className="admin-card admin-section">
            <h2 className="admin-section__title">Reports filed</h2>
            <TrendChart data={data.reports_filed_trend} color="var(--adm-red)" />
          </div>
        </>
      )}
    </section>
  );
}
