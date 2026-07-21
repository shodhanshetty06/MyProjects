/**
 * Tracking feature - type contracts, mirroring
 * backend/app/modules/tracking/schemas.py.
 */
export type ApplicationStatus =
  | "applied"
  | "interview"
  | "assessment"
  | "rejected"
  | "offer"
  | "accepted"
  | "withdrawn";

export type ApplicationSource = "manual" | "auto_apply";

export type ApplicationSortBy = "applied_at" | "updated_at" | "company_name" | "job_title" | "status";
export type SortDirection = "asc" | "desc";

export interface ApplicationRead {
  id: string;
  user_id: string;
  job_id: string | null;
  company_name: string;
  job_title: string;
  job_url: string | null;
  location: string | null;
  status: ApplicationStatus;
  source: ApplicationSource;
  applied_at: string;
  created_at: string;
  updated_at: string;
}

export interface ApplicationNoteRead {
  id: string;
  content: string;
  created_at: string;
  updated_at: string;
}

export interface ApplicationStatusHistoryRead {
  id: string;
  from_status: ApplicationStatus | null;
  to_status: ApplicationStatus;
  note: string | null;
  created_at: string;
}

export interface ApplicationDetail extends ApplicationRead {
  notes: ApplicationNoteRead[];
  status_history: ApplicationStatusHistoryRead[];
}

export interface TimelineEntry {
  type: "status_change" | "note";
  occurred_at: string;
  data: ApplicationStatusHistoryRead | ApplicationNoteRead;
}

export interface ApplicationStatistics {
  total_applications: number;
  status_counts: Partial<Record<ApplicationStatus, number>>;
  applications_last_7_days: number;
  applications_last_30_days: number;
  interview_rate: number;
  offer_rate: number;
  acceptance_rate: number;
  average_days_to_first_response: number | null;
}

export interface DashboardResponse {
  statistics: ApplicationStatistics;
  recent_applications: ApplicationRead[];
  needs_follow_up: ApplicationRead[];
}

export interface ApplicationCreateRequest {
  job_id?: string | null;
  company_name: string;
  job_title: string;
  job_url?: string | null;
  location?: string | null;
  source?: ApplicationSource;
  status?: ApplicationStatus;
  applied_at?: string;
}

export interface ApplicationUpdateRequest {
  company_name?: string;
  job_title?: string;
  job_url?: string | null;
  location?: string | null;
  applied_at?: string;
}

export interface ApplicationStatusChangeRequest {
  status: ApplicationStatus;
  note?: string;
}

export interface ApplicationFiltersParams {
  q?: string;
  status?: ApplicationStatus[];
  company_name?: string;
  job_id?: string;
  applied_after?: string;
  applied_before?: string;
  sort_by?: ApplicationSortBy;
  sort_direction?: SortDirection;
  page?: number;
  page_size?: number;
}

export const APPLICATION_STATUSES: ApplicationStatus[] = [
  "applied",
  "assessment",
  "interview",
  "offer",
  "accepted",
  "rejected",
  "withdrawn",
];
