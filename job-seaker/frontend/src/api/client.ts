import axios, { type AxiosError, type AxiosInstance, type InternalAxiosRequestConfig } from "axios";

import { config } from "@/app/config";
import { getAccessToken, setAccessToken } from "@/features/auth/tokenStore";
import type { AccessTokenResponse } from "@/features/auth/types";
import type { BaseResponse, ErrorResponse } from "@/types/common";

/**
 * Shared Axios instance every feature's api.ts should import instead of
 * calling axios directly, so base URL, headers, and error normalization
 * stay consistent across modules.
 */
export const apiClient: AxiosInstance = axios.create({
  baseURL: config.apiBaseUrl,
  // Carries the httpOnly refresh-token cookie (scoped to /auth by the
  // backend) — never read from JS, only replayed by the browser.
  withCredentials: true,
  headers: {
    "Content-Type": "application/json",
  },
});

apiClient.interceptors.request.use((requestConfig) => {
  const token = getAccessToken();
  if (token) {
    requestConfig.headers.set("Authorization", `Bearer ${token}`);
  }
  return requestConfig;
});

/**
 * POSTs to `/auth/refresh` using a bare axios call (not `apiClient`) so it
 * can't recursively trigger the response interceptor below. Exported so
 * `AuthProvider` reuses the exact same call for the initial-load silent
 * login instead of duplicating it.
 */
export async function refreshAccessToken(): Promise<string | null> {
  try {
    const response = await axios.post<BaseResponse<AccessTokenResponse>>(
      `${config.apiBaseUrl}/auth/refresh`,
      undefined,
      { withCredentials: true },
    );
    const token = response.data.data?.access_token ?? null;
    setAccessToken(token);
    return token;
  } catch {
    setAccessToken(null);
    return null;
  }
}

let pendingRefresh: Promise<string | null> | null = null;

function isAuthEndpoint(url: string | undefined): boolean {
  return !!url && /\/auth\/(login|register|refresh)$/.test(url);
}

apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError<ErrorResponse>) => {
    const originalRequest = error.config as
      | (InternalAxiosRequestConfig & { _retried?: boolean })
      | undefined;

    const shouldAttemptRefresh =
      error.response?.status === 401 &&
      originalRequest &&
      !originalRequest._retried &&
      !isAuthEndpoint(originalRequest.url);

    if (shouldAttemptRefresh) {
      originalRequest._retried = true;
      pendingRefresh ??= refreshAccessToken().finally(() => {
        pendingRefresh = null;
      });
      const token = await pendingRefresh;
      if (token) {
        originalRequest.headers.set("Authorization", `Bearer ${token}`);
        return apiClient(originalRequest);
      }
    }

    return Promise.reject(error);
  },
);
