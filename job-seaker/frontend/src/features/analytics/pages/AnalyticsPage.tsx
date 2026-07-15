import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import "@/features/analytics/analytics.css";
import { exportApplicationsCsv, exportDashboardPdf, getAnalyticsDashboard } from "@/features/analytics/api";
import { BarListChart } from "@/features/analytics/components/BarListChart";
import { ChartCard } from "@/features/analytics/components/ChartCard";
import { ColumnChart } from "@/features/analytics/components/ColumnChart";
import { DataTable } from "@/features/analytics/components/DataTable";
import { FilterBar } from "@/features/analytics/components/FilterBar";
import { HeatmapCalendar } from "@/features/analytics/components/HeatmapCalendar";
import { StatTile } from "@/features/analytics/components/StatTile";
import { TrendLineChart } from "@/features/analytics/components/TrendLineChart";
import { formatCompactNumber, formatDays, formatPercent } from "@/features/analytics/format";
import { categorical, pick, useColorScheme } from "@/features/analytics/theme";
import type { AnalyticsFiltersParams, CompanyOutcome, HiringCompanyCount, TrendInterval } from "@/features/analytics/types";

function isoDaysAgo(days: number): string {
  const date = new Date();
  date.setUTCDate(date.getUTCDate() - days);
  return date.toISOString();
}

export function AnalyticsPage() {
  const scheme = useColorScheme();
  const primary = pick(categorical.blue, scheme);
  const secondary = pick(categorical.aqua, scheme);

  const [interval, setInterval] = useState<TrendInterval>("week");
  const [companyName, setCompanyName] = useState("");
  const [activePresetDays, setActivePresetDays] = useState<number | null>(90);
  const [dateFrom, setDateFrom] = useState<string | undefined>(() => isoDaysAgo(90));
  const [dateTo, setDateTo] = useState<string | undefined>(undefined);
  const [isExportingCsv, setIsExportingCsv] = useState(false);
  const [isExportingPdf, setIsExportingPdf] = useState(false);

  const filters: AnalyticsFiltersParams = useMemo(
    () => ({
      interval,
      company_name: companyName.trim() || undefined,
      date_from: dateFrom,
      date_to: dateTo,
      top_n: 10,
    }),
    [interval, companyName, dateFrom, dateTo],
  );

  const { data, isLoading, isFetching, isError } = useQuery({
    queryKey: ["analytics", "dashboard", filters],
    queryFn: () => getAnalyticsDashboard(filters),
    placeholderData: keepPreviousData,
  });

  const handlePresetSelect = (days: number) => {
    setActivePresetDays(days);
    setDateFrom(isoDaysAgo(days));
    setDateTo(undefined);
  };

  const handleCustomRangeChange = (from: string | undefined, to: string | undefined) => {
    setActivePresetDays(null);
    setDateFrom(from);
    setDateTo(to);
  };

  const handleExportCsv = async () => {
    setIsExportingCsv(true);
    try {
      await exportApplicationsCsv(filters);
    } finally {
      setIsExportingCsv(false);
    }
  };

  const handleExportPdf = async () => {
    setIsExportingPdf(true);
    try {
      await exportDashboardPdf(filters);
    } finally {
      setIsExportingPdf(false);
    }
  };

  return (
    <section className="analytics-page">
      <header className="analytics-page__header">
        <h1>Analytics</h1>
        <p className="analytics-page__subtitle">Your job search, measured.</p>
      </header>

      <FilterBar
        filters={filters}
        activePresetDays={activePresetDays}
        onPresetSelect={handlePresetSelect}
        onCustomRangeChange={handleCustomRangeChange}
        onIntervalChange={setInterval}
        onCompanyNameChange={setCompanyName}
        onExportCsv={handleExportCsv}
        onExportPdf={handleExportPdf}
        isExportingCsv={isExportingCsv}
        isExportingPdf={isExportingPdf}
      />

      {isLoading && <p className="analytics-status">Loading analytics…</p>}
      {isError && <p className="analytics-status analytics-status--error">Unable to load analytics.</p>}

      {data && (
        <div className="analytics-content" style={{ opacity: isFetching ? 0.6 : 1 }}>
          <div className="analytics-summary">
            <StatTile label="Total applications" value={formatCompactNumber(data.summary.total_applications)} />
            <StatTile label="Interview rate" value={formatPercent(data.summary.interview_rate)} />
            <StatTile label="Offer rate" value={formatPercent(data.summary.offer_rate)} />
            <StatTile label="Acceptance rate" value={formatPercent(data.summary.acceptance_rate)} />
            <StatTile label="Avg. response time" value={formatDays(data.summary.average_response_days)} />
          </div>

          <div className="analytics-grid">
            <ChartCard title="Applications" subtitle="Applications submitted over time">
              <TrendLineChart
                interval={data.applications_trend.interval}
                data={data.applications_trend.points}
                series={[{ key: "count", label: "Applications", color: primary }]}
              />
            </ChartCard>

            <ChartCard title="Interviews" subtitle="Applications that reached an interview or assessment">
              <TrendLineChart
                interval={data.interviews_trend.interval}
                data={data.interviews_trend.points}
                series={[{ key: "count", label: "Interviews", color: primary }]}
              />
            </ChartCard>

            <ChartCard title="Offers" subtitle="Offers received over time">
              <TrendLineChart
                interval={data.offers_trend.interval}
                data={data.offers_trend.points}
                series={[{ key: "count", label: "Offers", color: primary }]}
              />
            </ChartCard>

            <ChartCard
              title="Acceptance rate"
              subtitle={`${data.acceptance_rate.overall_accepted} accepted of ${data.acceptance_rate.overall_offers} offers`}
            >
              <TrendLineChart
                interval={data.acceptance_rate.interval}
                data={data.acceptance_rate.points}
                series={[{ key: "acceptance_rate", label: "Acceptance rate", color: primary }]}
                valueFormatter={(value) => formatPercent(value)}
              />
            </ChartCard>

            <ChartCard title="Resume performance" subtitle="How your base resumes are trending">
              <div className="analytics-substats">
                <StatTile label="Resumes" value={String(data.resume_performance.total_resumes)} />
                <StatTile label="Versions" value={String(data.resume_performance.total_versions)} />
                <StatTile
                  label="Parse success"
                  value={formatPercent(data.resume_performance.parse_success_rate)}
                />
                <StatTile
                  label="Avg. skills / resume"
                  value={data.resume_performance.average_skills_per_version.toFixed(1)}
                />
              </div>
              <TrendLineChart
                interval={interval}
                data={data.resume_performance.versions_trend}
                series={[{ key: "count", label: "Versions uploaded", color: primary }]}
              />
            </ChartCard>

            <ChartCard title="ATS resume performance" subtitle="Optimization score trends across your ATS-tailored resumes">
              <div className="analytics-substats">
                <StatTile
                  label="Avg. ATS score"
                  value={
                    data.ats_resume_performance.average_ats_score !== null
                      ? data.ats_resume_performance.average_ats_score.toFixed(0)
                      : "N/A"
                  }
                />
                <StatTile
                  label="Avg. keyword score"
                  value={
                    data.ats_resume_performance.average_keyword_score !== null
                      ? data.ats_resume_performance.average_keyword_score.toFixed(0)
                      : "N/A"
                  }
                />
                <StatTile
                  label="Completion rate"
                  value={formatPercent(data.ats_resume_performance.completion_rate)}
                />
              </div>
              <TrendLineChart
                interval={interval}
                data={data.ats_resume_performance.score_trend}
                series={[
                  { key: "average_ats_score", label: "ATS score", color: primary },
                  { key: "average_keyword_score", label: "Keyword score", color: secondary },
                ]}
              />
              <ColumnChart
                data={data.ats_resume_performance.score_distribution.map((bucket) => ({
                  label: bucket.range_label,
                  value: bucket.count,
                }))}
                color={secondary}
                height={140}
              />
            </ChartCard>

            <ChartCard title="Top skills" subtitle="From your current resume(s)">
              <BarListChart
                items={data.top_skills.map((s) => ({ label: s.skill, value: s.count }))}
                color={primary}
              />
            </ChartCard>

            <ChartCard title="Top companies" subtitle="Where you've applied">
              <DataTable<CompanyOutcome>
                rowKey={(row) => row.company_name}
                rows={data.top_companies}
                columns={[
                  { key: "company", header: "Company", render: (row) => row.company_name },
                  { key: "applications", header: "Applications", align: "right", render: (row) => row.applications },
                  { key: "interviews", header: "Interviews", align: "right", render: (row) => row.interviews },
                  { key: "offers", header: "Offers", align: "right", render: (row) => row.offers },
                ]}
              />
            </ChartCard>

            <ChartCard title="Top countries" subtitle="Where your applications are located">
              <BarListChart
                items={data.top_countries.map((c) => ({ label: c.country, value: c.count }))}
                color={primary}
              />
            </ChartCard>

            <ChartCard title="Hiring trends" subtitle="Market-wide job postings ingested over time">
              <TrendLineChart
                interval={data.hiring_trends.interval}
                data={data.hiring_trends.trend}
                series={[{ key: "jobs_posted", label: "Jobs posted", color: primary }]}
              />
              <DataTable<HiringCompanyCount>
                rowKey={(row) => row.company_name}
                rows={data.hiring_trends.top_hiring_companies}
                columns={[
                  { key: "company", header: "Company", render: (row) => row.company_name },
                  { key: "jobs", header: "Jobs posted", align: "right", render: (row) => row.jobs_posted },
                ]}
              />
            </ChartCard>

            <ChartCard title="Skill demand" subtitle="Most in-demand skills across active job postings">
              <BarListChart
                items={data.skill_demand.map((s) => ({
                  label: s.skill,
                  value: s.job_count,
                  badge: s.user_has_skill ? "✓ you have this" : undefined,
                }))}
                color={primary}
              />
            </ChartCard>

            <ChartCard title="Response time" subtitle="Days from applying to the first status update">
              <div className="analytics-substats">
                <StatTile label="Overall average" value={formatDays(data.response_time.overall_average_days)} />
              </div>
              <TrendLineChart
                interval={data.response_time.interval}
                data={data.response_time.points}
                series={[{ key: "average_days", label: "Avg. response days", color: primary }]}
                valueFormatter={(value) => value.toFixed(1)}
              />
            </ChartCard>

            <ChartCard title="Application heatmap" subtitle="Your application activity by day">
              <div className="analytics-substats">
                <StatTile label="Total applications" value={String(data.heatmap.total)} />
                <StatTile label="Current streak" value={`${data.heatmap.current_streak_days}d`} />
                <StatTile label="Longest streak" value={`${data.heatmap.longest_streak_days}d`} />
              </div>
              <HeatmapCalendar cells={data.heatmap.cells} />
            </ChartCard>
          </div>
        </div>
      )}
    </section>
  );
}
