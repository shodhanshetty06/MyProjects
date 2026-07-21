import { apiClient } from "@/api/client";
import type {
  GmailAccountRead,
  GmailConnectResponse,
  TrackedEmailDetail,
  TrackedEmailFiltersParams,
  TrackedEmailRead,
  TrackedEmailResolveRequest,
} from "@/features/email-tracking/types";
import type { BaseResponse, PaginatedResponse } from "@/types/common";

/** Email Tracking feature — API bindings, `X-User-Id` trusted header convention. */
function authHeaders(userId: string) {
  return { headers: { "X-User-Id": userId } };
}

export async function getGmailConnectUrl(userId: string): Promise<string> {
  const response = await apiClient.get<BaseResponse<GmailConnectResponse>>(
    "/email-tracking/gmail/connect",
    authHeaders(userId),
  );
  return (response.data.data as GmailConnectResponse).authorization_url;
}

export async function listGmailAccounts(userId: string): Promise<GmailAccountRead[]> {
  const response = await apiClient.get<BaseResponse<GmailAccountRead[]>>(
    "/email-tracking/accounts",
    authHeaders(userId),
  );
  return response.data.data ?? [];
}

export async function disconnectGmailAccount(userId: string, accountId: string): Promise<void> {
  await apiClient.delete(`/email-tracking/accounts/${accountId}`, authHeaders(userId));
}

export async function listTrackedEmails(
  userId: string,
  filters: TrackedEmailFiltersParams = {},
): Promise<PaginatedResponse<TrackedEmailRead>> {
  const response = await apiClient.get<PaginatedResponse<TrackedEmailRead>>("/email-tracking/emails", {
    ...authHeaders(userId),
    params: filters,
  });
  return response.data;
}

export async function getTrackedEmail(userId: string, emailId: string): Promise<TrackedEmailDetail> {
  const response = await apiClient.get<BaseResponse<TrackedEmailDetail>>(
    `/email-tracking/emails/${emailId}`,
    authHeaders(userId),
  );
  return response.data.data as TrackedEmailDetail;
}

export async function resolveTrackedEmail(
  userId: string,
  emailId: string,
  payload: TrackedEmailResolveRequest,
): Promise<TrackedEmailRead> {
  const response = await apiClient.post<BaseResponse<TrackedEmailRead>>(
    `/email-tracking/emails/${emailId}/resolve`,
    payload,
    authHeaders(userId),
  );
  return response.data.data as TrackedEmailRead;
}
