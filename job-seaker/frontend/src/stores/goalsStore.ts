import { create } from "zustand";
import { persist } from "zustand/middleware";

interface GoalsState {
  weeklyApplicationGoal: number;
  setWeeklyApplicationGoal: (goal: number) => void;
}

/**
 * The goal target is a user preference with no backend field to live in -
 * progress against it is computed from real tracking-module data
 * (applications_last_7_days) wherever this is read.
 */
export const useGoalsStore = create<GoalsState>()(
  persist(
    (set) => ({
      weeklyApplicationGoal: 10,
      setWeeklyApplicationGoal: (goal) => set({ weeklyApplicationGoal: Math.max(1, Math.round(goal)) }),
    }),
    { name: "career-goals" },
  ),
);
