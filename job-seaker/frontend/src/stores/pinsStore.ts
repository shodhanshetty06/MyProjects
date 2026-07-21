import { create } from "zustand";
import { persist } from "zustand/middleware";

export type PinnedItemType = "job" | "resume" | "company";

export interface PinnedItem {
  id: string;
  type: PinnedItemType;
  label: string;
  meta?: string;
  pinnedAt: string;
}

interface PinsState {
  items: PinnedItem[];
  isPinned: (type: PinnedItemType, id: string) => boolean;
  togglePin: (item: Omit<PinnedItem, "pinnedAt">) => void;
  unpin: (type: PinnedItemType, id: string) => void;
}

/**
 * Local, per-browser "save for later" concept. The backend has no
 * save/pin endpoint for jobs, resumes, or companies (recommendations only
 * support dismiss) - this is intentionally client-only, not synced.
 */
export const usePinsStore = create<PinsState>()(
  persist(
    (set, get) => ({
      items: [],
      isPinned: (type, id) => get().items.some((item) => item.type === type && item.id === id),
      togglePin: (item) =>
        set((state) => {
          const exists = state.items.some((i) => i.type === item.type && i.id === item.id);
          if (exists) {
            return { items: state.items.filter((i) => !(i.type === item.type && i.id === item.id)) };
          }
          return { items: [{ ...item, pinnedAt: new Date().toISOString() }, ...state.items] };
        }),
      unpin: (type, id) =>
        set((state) => ({ items: state.items.filter((i) => !(i.type === type && i.id === id)) })),
    }),
    { name: "pinned-items" },
  ),
);
