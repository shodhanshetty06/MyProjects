import { apiClient } from "@/api/client";
import type {
  OnboardingCompletePayload,
  OnboardingCompleteResponse,
  OnboardingStatusRead,
} from "@/features/onboarding/types";
import { paths } from "@/routes/paths";
import type { BaseResponse } from "@/types/common";

/**
 * Onboarding feature — API bindings. Unlike most other feature modules,
 * these calls carry no `X-User-Id` header: the caller's identity comes
 * from the JWT bearer token `apiClient`'s request interceptor already
 * attaches (see `app/api/deps.py::get_current_user_id`).
 */

export async function getOnboardingStatus(): Promise<OnboardingStatusRead> {
  const response = await apiClient.get<BaseResponse<OnboardingStatusRead>>("/onboarding/status");
  return response.data.data as OnboardingStatusRead;
}

export async function completeOnboarding(
  payload: OnboardingCompletePayload,
): Promise<OnboardingCompleteResponse> {
  const response = await apiClient.post<BaseResponse<OnboardingCompleteResponse>>(
    "/onboarding/complete",
    payload,
  );
  return response.data.data as OnboardingCompleteResponse;
}

/**
 * Called right after a successful login/register so the caller knows
 * whether to land on `/onboarding` or `/workspace`. Fails open to
 * `/onboarding` — an onboarding-status hiccup shouldn't strand a fresh
 * session on a blank redirect.
 */
export async function resolvePostLoginPath(): Promise<string> {
  try {
    const status = await getOnboardingStatus();
    return status.onboarding_completed ? paths.workspace : paths.onboarding;
  } catch {
    return paths.onboarding;
  }
}
