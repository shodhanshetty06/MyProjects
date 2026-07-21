import { apiClient } from "@/api/client";
import type {
  NotificationPreferenceRead,
  NotificationPreferenceUpdate,
  NotificationRead,
} from "@/features/notification/types";
import type { BaseResponse, PaginatedResponse } from "@/types/common";

/** Notification feature — API bindings, `X-User-Id` trusted header convention. */
function authHeaders(userId: string) {
  return { headers: { "X-User-Id": userId } };
}

export async function listNotifications(
  userId: string,
  unreadOnly = false,
  page = 1,
  pageSize = 30,
): Promise<PaginatedResponse<NotificationRead>> {
  const response = await apiClient.get<PaginatedResponse<NotificationRead>>("/notifications", {
    ...authHeaders(userId),
    params: { unread_only: unreadOnly, page, page_size: pageSize },
  });
  return response.data;
}

export async function markNotificationRead(userId: string, notificationId: string): Promise<NotificationRead> {
  const response = await apiClient.post<BaseResponse<NotificationRead>>(
    `/notifications/${notificationId}/read`,
    undefined,
    authHeaders(userId),
  );
  return response.data.data as NotificationRead;
}

export async function markAllNotificationsRead(userId: string): Promise<number> {
  const response = await apiClient.post<BaseResponse<{ marked_read: number }>>(
    "/notifications/read-all",
    undefined,
    authHeaders(userId),
  );
  return response.data.data?.marked_read ?? 0;
}

export async function getNotificationPreferences(userId: string): Promise<NotificationPreferenceRead[]> {
  const response = await apiClient.get<BaseResponse<NotificationPreferenceRead[]>>(
    "/notifications/preferences",
    authHeaders(userId),
  );
  return response.data.data ?? [];
}

export async function updateNotificationPreferences(
  userId: string,
  preferences: NotificationPreferenceUpdate[],
): Promise<NotificationPreferenceRead[]> {
  const response = await apiClient.put<BaseResponse<NotificationPreferenceRead[]>>(
    "/notifications/preferences",
    { preferences },
    authHeaders(userId),
  );
  return response.data.data ?? [];
}

/**
 * Phase 0 module-connectivity stub, kept only because nothing has removed
 * its last caller yet.
 */
export async function getNotificationModuleStatus(): Promise<BaseResponse> {
  const response = await apiClient.get<BaseResponse>("/notifications/status");
  return response.data;
}
