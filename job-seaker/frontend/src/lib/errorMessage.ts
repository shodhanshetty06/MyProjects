import type { AxiosError } from "axios";

import type { ErrorResponse } from "@/types/common";

export function errorMessage(error: unknown, fallback = "Something went wrong. Please try again."): string {
  const response = (error as AxiosError<ErrorResponse> | undefined)?.response;
  return response?.data?.error?.message ?? fallback;
}
