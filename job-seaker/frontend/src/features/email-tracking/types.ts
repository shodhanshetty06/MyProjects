/**
 * Email Tracking feature - type contracts, mirroring
 * backend/app/modules/email_tracking/schemas.py. Surfaced in the app as a
 * "Tracked Emails" tab within the Application Tracker.
 */
export type GmailAccountStatus = "connected" | "needs_reauth" | "disconnected";
export type EmailClassification = "interview" | "assessment" | "rejection" | "offer" | "not_relevant";
export type EmailMatchStatus = "matched" | "needs_review" | "unmatched" | "ignored";

export interface GmailConnectResponse {
  authorization_url: string;
}

export interface GmailAccountRead {
  id: string;
  email_address: string;
  status: GmailAccountStatus;
  history_id: string | null;
  last_synced_at: string | null;
  last_sync_error: string | null;
  created_at: string;
}

export interface TrackedEmailRead {
  id: string;
  gmail_account_id: string;
  application_id: string | null;
  sender: string;
  subject: string;
  snippet: string;
  received_at: string;
  classification: EmailClassification;
  confidence: number;
  match_status: EmailMatchStatus;
  created_at: string;
}

export interface TrackedEmailDetail extends TrackedEmailRead {
  matched_status_history_id: string | null;
  raw_metadata: Record<string, unknown>;
}

export interface TrackedEmailResolveRequest {
  dismiss?: boolean;
  application_id?: string;
  status?: string;
}

export interface TrackedEmailFiltersParams {
  match_status?: EmailMatchStatus;
  application_id?: string;
  page?: number;
  page_size?: number;
}
