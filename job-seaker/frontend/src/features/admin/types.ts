/**
 * Admin feature — type contracts.
 *
 * Mirrors backend/app/modules/admin/schemas.py.
 */

export type AdminRole = "super_admin" | "admin" | "moderator" | "support" | "analyst";

export type Permission =
  | "dashboard.view"
  | "users.manage"
  | "companies.verify"
  | "jobs.moderate"
  | "content.moderate"
  | "reports.manage"
  | "feature_flags.manage"
  | "audit_logs.view"
  | "system_logs.view"
  | "analytics.view"
  | "permissions.manage";

export type UserAccountStatus = "active" | "suspended" | "banned";
export type VerificationStatus = "unverified" | "pending" | "verified" | "rejected";
export type ModerationStatus = "active" | "flagged" | "removed" | "pending_review";
export type ReportTargetType = "job" | "company" | "user";
export type ReportStatus = "open" | "reviewing" | "resolved" | "dismissed";
export type JobStatus = "pending" | "normalizing" | "active" | "failed" | "closed";
export type ApplicationStatus =
  | "applied"
  | "interview"
  | "assessment"
  | "rejected"
  | "offer"
  | "accepted"
  | "withdrawn";

export interface PaginationParams {
  page?: number;
  page_size?: number;
}

// --- actors / permissions --------------------------------------------------

export interface AdminActorBootstrapRequest {
  email: string;
  display_name?: string;
}

export interface AdminActorCreateRequest {
  user_id: string;
  email: string;
  display_name: string;
  role: AdminRole;
}

export interface AdminActorUpdateRequest {
  display_name?: string;
  role?: AdminRole;
  is_active?: boolean;
}

export interface AdminActorResponse {
  id: string;
  user_id: string;
  email: string;
  display_name: string;
  role: AdminRole;
  is_active: boolean;
  last_active_at: string | null;
  created_at: string;
}

export interface RolePermissions {
  role: AdminRole;
  permissions: Permission[];
}

export interface PermissionsResponse {
  current_role: AdminRole;
  current_permissions: Permission[];
  roles: RolePermissions[];
}

// --- dashboard ---------------------------------------------------------------

export interface AuditLogEntry {
  id: string;
  actor_id: string | null;
  actor_email: string | null;
  action: string;
  target_type: string;
  target_id: string | null;
  before: Record<string, unknown> | null;
  after: Record<string, unknown> | null;
  notes: string | null;
  created_at: string;
}

export interface SystemLogEntry {
  source: string;
  severity: string;
  message: string;
  target_id: string | null;
  occurred_at: string;
}

export interface DashboardResponse {
  generated_at: string;
  users_total: number;
  users_active: number;
  users_suspended: number;
  users_banned: number;
  companies_total: number;
  companies_verified: number;
  companies_pending: number;
  jobs_total: number;
  jobs_active: number;
  jobs_flagged: number;
  applications_total: number;
  reports_open: number;
  feature_flags_enabled: number;
  recent_audit_logs: AuditLogEntry[];
  recent_system_logs: SystemLogEntry[];
}

// --- users ---------------------------------------------------------------

export interface UserFiltersParams extends PaginationParams {
  search?: string;
  status?: UserAccountStatus;
}

export interface UserSummary {
  user_id: string;
  status: UserAccountStatus;
  application_count: number;
  first_seen_at: string | null;
  last_activity_at: string | null;
}

export interface UserDetail extends UserSummary {
  reason: string | null;
  changed_at: string | null;
  applications_by_status: Record<string, number>;
  auto_apply_task_count: number;
  notification_count: number;
}

export interface UserStatusUpdateRequest {
  status: UserAccountStatus;
  reason?: string;
}

// --- companies --------------------------------------------------------------

export interface CompanyFiltersParams extends PaginationParams {
  search?: string;
  verification_status?: VerificationStatus;
  is_active?: boolean;
}

export interface CompanySummary {
  id: string;
  name: string;
  website: string | null;
  industry: string | null;
  is_active: boolean;
  verification_status: VerificationStatus;
  jobs_count: number;
  created_at: string;
}

export interface CompanyDetail extends CompanySummary {
  description: string | null;
  logo_url: string | null;
  notes: string | null;
  reviewed_by: string | null;
  reviewed_at: string | null;
}

export interface CompanyVerificationUpdateRequest {
  status: VerificationStatus;
  notes?: string;
}

// --- jobs ---------------------------------------------------------------

export interface JobFiltersParams extends PaginationParams {
  search?: string;
  moderation_status?: ModerationStatus;
  job_status?: JobStatus;
  company_id?: string;
}

export interface JobSummary {
  id: string;
  title: string;
  company_id: string;
  company_name: string;
  status: JobStatus;
  moderation_status: ModerationStatus;
  flag_count: number;
  posted_at: string | null;
  created_at: string;
}

export interface JobModerationUpdateRequest {
  status: ModerationStatus;
  reason?: string;
}

// --- applications ---------------------------------------------------------

export interface ApplicationFiltersParams extends PaginationParams {
  search?: string;
  user_id?: string;
  status?: ApplicationStatus;
}

export interface ApplicationSummary {
  id: string;
  user_id: string;
  company_name: string;
  job_title: string;
  status: ApplicationStatus;
  source: string;
  applied_at: string;
}

// --- reports / content moderation -------------------------------------------

export interface ReportFiltersParams extends PaginationParams {
  status?: ReportStatus;
  target_type?: ReportTargetType;
}

export interface ReportCreateRequest {
  target_type: ReportTargetType;
  target_id: string;
  reason: string;
}

export interface ReportResponse {
  id: string;
  reporter_user_id: string;
  target_type: ReportTargetType;
  target_id: string;
  reason: string;
  status: ReportStatus;
  resolved_by: string | null;
  resolution_notes: string | null;
  resolved_at: string | null;
  created_at: string;
}

export interface ReportUpdateRequest {
  status: ReportStatus;
  resolution_notes?: string;
}

export interface ModerationQueueItem {
  kind: "report" | "job" | "company";
  id: string;
  target_type: string;
  target_id: string;
  summary: string;
  status: string;
  created_at: string;
}

export interface ModerationQueueResponse {
  items: ModerationQueueItem[];
  total_open: number;
}

// --- feature flags ------------------------------------------------------------

export interface FeatureFlagCreateRequest {
  key: string;
  name: string;
  description?: string;
  is_enabled?: boolean;
  rollout_percentage?: number;
}

export interface FeatureFlagUpdateRequest {
  name?: string;
  description?: string;
  is_enabled?: boolean;
  rollout_percentage?: number;
}

export interface FeatureFlagResponse {
  id: string;
  key: string;
  name: string;
  description: string | null;
  is_enabled: boolean;
  rollout_percentage: number;
  updated_by: string | null;
  created_at: string;
  updated_at: string;
}

// --- audit logs / system logs ------------------------------------------------

export interface AuditLogFiltersParams extends PaginationParams {
  actor_id?: string;
  action?: string;
  target_type?: string;
  date_from?: string;
  date_to?: string;
}

export interface SystemLogFiltersParams extends PaginationParams {
  source?: string;
  severity?: string;
  date_from?: string;
  date_to?: string;
}

// --- platform analytics --------------------------------------------------------

export interface PlatformTrendPoint {
  bucket: string;
  count: number;
}

export interface PlatformAnalyticsResponse {
  generated_at: string;
  days: number;
  active_users_trend: PlatformTrendPoint[];
  applications_trend: PlatformTrendPoint[];
  jobs_posted_trend: PlatformTrendPoint[];
  companies_added_trend: PlatformTrendPoint[];
  reports_filed_trend: PlatformTrendPoint[];
  reports_resolved_count: number;
  jobs_moderated_count: number;
  companies_reviewed_count: number;
}
