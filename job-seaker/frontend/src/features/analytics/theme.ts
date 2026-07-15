/**
 * Analytics dashboard chart theme — the validated reference palette (see the
 * `dataviz` skill's `references/palette.md`). Categorical hues are assigned
 * in this fixed order everywhere in the dashboard; never cycled or reused
 * for a different series.
 */
import { useEffect, useState } from "react";

export type ColorScheme = "light" | "dark";

export function useColorScheme(): ColorScheme {
  const [scheme, setScheme] = useState<ColorScheme>(() =>
    window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light",
  );

  useEffect(() => {
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = (event: MediaQueryListEvent) => setScheme(event.matches ? "dark" : "light");
    query.addEventListener("change", handleChange);
    return () => query.removeEventListener("change", handleChange);
  }, []);

  return scheme;
}

export function pick(pair: { light: string; dark: string }, scheme: ColorScheme): string {
  return scheme === "dark" ? pair.dark : pair.light;
}

export const categorical = {
  blue: { light: "#2a78d6", dark: "#3987e5" },
  aqua: { light: "#1baf7a", dark: "#199e70" },
  yellow: { light: "#eda100", dark: "#c98500" },
  green: { light: "#008300", dark: "#008300" },
  violet: { light: "#4a3aa7", dark: "#9085e9" },
  red: { light: "#e34948", dark: "#e66767" },
} as const;

export const sequentialBlue = {
  100: "#cde2fb",
  200: "#9ec5f4",
  300: "#6da7ec",
  400: "#3987e5",
  500: "#256abf",
  600: "#184f95",
  700: "#0d366b",
} as const;

export const chrome = {
  surface: { light: "#fcfcfb", dark: "#1a1a19" },
  primaryInk: { light: "#0b0b0b", dark: "#ffffff" },
  secondaryInk: { light: "#52514e", dark: "#c3c2b7" },
  mutedInk: { light: "#898781", dark: "#898781" },
  gridline: { light: "#e1e0d9", dark: "#2c2c2a" },
  baseline: { light: "#c3c2b7", dark: "#383835" },
} as const;

/** Series-1 (blue) is the default for every single-series chart. */
export const primarySeries = categorical.blue;
export const secondarySeries = categorical.aqua;
