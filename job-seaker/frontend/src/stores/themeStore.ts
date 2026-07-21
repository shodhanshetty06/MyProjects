import { create } from "zustand";
import { persist } from "zustand/middleware";

export type ThemePreference = "light" | "dark" | "system";

interface ThemeState {
  preference: ThemePreference;
  setPreference: (preference: ThemePreference) => void;
}

function resolveTheme(preference: ThemePreference): "light" | "dark" {
  if (preference === "system") {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  return preference;
}

function applyTheme(preference: ThemePreference) {
  document.documentElement.setAttribute("data-theme", resolveTheme(preference));
}

export const useThemeStore = create<ThemeState>()(
  persist(
    (set) => ({
      preference: "system",
      setPreference: (preference) => {
        set({ preference });
        applyTheme(preference);
      },
    }),
    {
      name: "theme-preference",
      onRehydrateStorage: () => (state) => {
        applyTheme(state?.preference ?? "system");
      },
    },
  ),
);

// Apply immediately on module load (before React mounts) so there's no
// light-mode flash for users whose persisted/system preference is dark.
applyTheme(useThemeStore.getState().preference);

if (typeof window !== "undefined") {
  window
    .matchMedia("(prefers-color-scheme: dark)")
    .addEventListener("change", () => {
      if (useThemeStore.getState().preference === "system") {
        applyTheme("system");
      }
    });
}
