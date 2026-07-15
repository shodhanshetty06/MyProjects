/**
 * Authentication feature — type contracts, mirroring
 * backend/app/modules/auth/schemas.py.
 */
export interface AuthUser {
  id: string;
  email: string;
  is_active: boolean;
  created_at: string;
}

export interface AccessTokenResponse {
  access_token: string;
  token_type: string;
  expires_at: string;
  user: AuthUser;
}

export interface RegisterPayload {
  email: string;
  password: string;
}

export interface LoginPayload {
  email: string;
  password: string;
}
