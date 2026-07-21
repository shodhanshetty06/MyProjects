export const config = {
  // Relative by default so calls go back through the nginx proxy at
  // whatever origin the page was loaded from (http://localhost:5000).
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL ?? "/api/v1",
} as const;
