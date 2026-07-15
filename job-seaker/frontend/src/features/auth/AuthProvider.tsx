import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { refreshAccessToken } from "@/api/client";
import { fetchCurrentUser, loginUser, logoutUser, registerUser } from "@/features/auth/api";
import { setAccessToken, subscribeToAccessToken } from "@/features/auth/tokenStore";
import type { AuthUser, LoginPayload, RegisterPayload } from "@/features/auth/types";

interface AuthContextValue {
  user: AuthUser | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (payload: LoginPayload) => Promise<void>;
  register: (payload: RegisterPayload) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // The token store has no concept of "user" — when a refresh fails (or the
  // user logs out) it nulls the token, and this keeps `user` in lockstep.
  useEffect(
    () =>
      subscribeToAccessToken((token) => {
        if (!token) setUser(null);
      }),
    [],
  );

  useEffect(() => {
    let cancelled = false;
    (async () => {
      // On first load there's no access token yet — the httpOnly refresh
      // cookie (if any, from a previous session) is the only signal, so
      // silently try to exchange it before deciding the user is logged out.
      const token = await refreshAccessToken();
      if (token && !cancelled) {
        try {
          const me = await fetchCurrentUser();
          if (!cancelled) setUser(me);
        } catch {
          if (!cancelled) setAccessToken(null);
        }
      }
      if (!cancelled) setIsLoading(false);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(async (payload: LoginPayload) => {
    const tokens = await loginUser(payload);
    setAccessToken(tokens.access_token);
    setUser(tokens.user);
  }, []);

  const register = useCallback(async (payload: RegisterPayload) => {
    const tokens = await registerUser(payload);
    setAccessToken(tokens.access_token);
    setUser(tokens.user);
  }, []);

  const logout = useCallback(async () => {
    try {
      await logoutUser();
    } finally {
      setAccessToken(null);
      setUser(null);
    }
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      isLoading,
      isAuthenticated: user !== null,
      login,
      register,
      logout,
    }),
    [user, isLoading, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return ctx;
}
