import { apiClient } from "@/api/client";
import type { AnalyticsDashboard, AnalyticsFiltersParams } from "@/features/analytics/types";
import type { BaseResponse } from "@/types/common";

/**
 * Analytics feature — API bindings.
 *
 * Mirrors backend/app/modules/analytics/router.py.
 */
export async function getAnalyticsDashboard(
  filters: AnalyticsFiltersParams,
): Promise<AnalyticsDashboard> {
  const response = await apiClient.get<BaseResponse<AnalyticsDashboard>>("/analytics/dashboard", {
    params: filters,
  });
  if (!response.data.data) {
    throw new Error("Analytics dashboard response had no data.");
  }
  return response.data.data;
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function filenameFromDisposition(disposition: string | undefined, fallback: string): string {
  const match = disposition?.match(/filename="?([^"]+)"?/);
  return match?.[1] ?? fallback;
}

export async function exportApplicationsCsv(filters: AnalyticsFiltersParams): Promise<void> {
  const response = await apiClient.get<Blob>("/analytics/export/csv", {
    params: filters,
    responseType: "blob",
  });
  const filename = filenameFromDisposition(response.headers["content-disposition"], "applications.csv");
  triggerDownload(response.data, filename);
}

export async function exportDashboardPdf(filters: AnalyticsFiltersParams): Promise<void> {
  const response = await apiClient.get<Blob>("/analytics/export/pdf", {
    params: filters,
    responseType: "blob",
  });
  const filename = filenameFromDisposition(
    response.headers["content-disposition"],
    "analytics-report.pdf",
  );
  triggerDownload(response.data, filename);
}
