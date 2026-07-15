import { apiClient } from "@/api/client";
import type {
  AccessTokenResponse,
  AuthUser,
  LoginPayload,
  RegisterPayload,
} from "@/features/auth/types";
import type { BaseResponse } from "@/types/common";

/** Authentication feature — API bindings. */
export async function registerUser(payload: RegisterPayload): Promise<AccessTokenResponse> {
  const response = await apiClient.post<BaseResponse<AccessTokenResponse>>(
    "/auth/register",
    payload,
  );
  return response.data.data as AccessTokenResponse;
}

export async function loginUser(payload: LoginPayload): Promise<AccessTokenResponse> {
  const response = await apiClient.post<BaseResponse<AccessTokenResponse>>("/auth/login", payload);
  return response.data.data as AccessTokenResponse;
}

export async function logoutUser(): Promise<void> {
  await apiClient.post("/auth/logout");
}

export async function fetchCurrentUser(): Promise<AuthUser> {
  const response = await apiClient.get<BaseResponse<AuthUser>>("/auth/me");
  return response.data.data as AuthUser;
}
