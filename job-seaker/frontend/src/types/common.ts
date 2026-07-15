/**
 * Mirrors backend/app/schemas/common.py — keep these in sync with the
 * BaseResponse / ErrorResponse / PaginatedResponse envelopes.
 */
export interface BaseResponse<T = unknown> {
  success: boolean;
  message: string;
  data: T | null;
  timestamp: string;
}

export interface ErrorDetail {
  error_code: string;
  message: string;
  details: Record<string, unknown>;
}

export interface ErrorResponse {
  success: false;
  error: ErrorDetail;
  request_id: string | null;
  timestamp: string;
}

export interface PaginationMeta {
  page: number;
  page_size: number;
  total_items: number;
  total_pages: number;
}

export interface PaginatedResponse<T = unknown> {
  success: boolean;
  message: string;
  data: T[];
  pagination: PaginationMeta;
  timestamp: string;
}
