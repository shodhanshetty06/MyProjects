/**
 * Notification feature - type contracts, mirroring
 * backend/app/modules/notification/schemas.py.
 */
export type NotificationEventType =
  | "application_status_changed"
  | "email_review_needed"
  | "gmail_reauth_required";

export type NotificationChannel = "email" | "push" | "in_app";
export type NotificationStatus = "pending" | "sent" | "failed";

export interface NotificationRead {
  id: string;
  event_type: NotificationEventType;
  channel: NotificationChannel;
  title: string;
  body: string;
  data: Record<string, unknown>;
  status: NotificationStatus;
  read_at: string | null;
  created_at: string;
}

export interface NotificationPreferenceRead {
  event_type: NotificationEventType;
  channel: NotificationChannel;
  enabled: boolean;
}

export interface NotificationPreferenceUpdate {
  event_type: NotificationEventType;
  channel: NotificationChannel;
  enabled: boolean;
}

export const NOTIFICATION_EVENT_TYPES: NotificationEventType[] = [
  "application_status_changed",
  "email_review_needed",
  "gmail_reauth_required",
];

export const NOTIFICATION_CHANNELS: NotificationChannel[] = ["email", "push", "in_app"];
