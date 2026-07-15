/**
 * Holds the in-memory access token outside React so `api/client.ts`'s axios
 * interceptors (which run before any component renders) can read/write it
 * without importing React state. `AuthProvider` is the only subscriber that
 * turns changes here into UI state.
 */
type Listener = (token: string | null) => void;

let accessToken: string | null = null;
const listeners = new Set<Listener>();

export function getAccessToken(): string | null {
  return accessToken;
}

export function setAccessToken(token: string | null): void {
  accessToken = token;
  listeners.forEach((listener) => listener(token));
}

export function subscribeToAccessToken(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
