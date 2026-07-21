import { create } from "zustand";

export type ToastTone = "default" | "success" | "warning" | "danger";

export interface Toast {
  id: string;
  title: string;
  description?: string;
  tone: ToastTone;
  duration: number;
}

interface ToastState {
  toasts: Toast[];
  push: (toast: Omit<Toast, "id" | "duration" | "tone"> & Partial<Pick<Toast, "duration" | "tone">>) => string;
  dismiss: (id: string) => void;
}

export const useToastStore = create<ToastState>()((set) => ({
  toasts: [],
  push: (toast) => {
    const id = crypto.randomUUID();
    set((state) => ({
      toasts: [...state.toasts, { id, tone: "default", duration: 5000, ...toast }],
    }));
    return id;
  },
  dismiss: (id) => set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),
}));

export const toast = {
  show: (title: string, description?: string) => useToastStore.getState().push({ title, description }),
  success: (title: string, description?: string) =>
    useToastStore.getState().push({ title, description, tone: "success" }),
  warning: (title: string, description?: string) =>
    useToastStore.getState().push({ title, description, tone: "warning" }),
  danger: (title: string, description?: string) =>
    useToastStore.getState().push({ title, description, tone: "danger" }),
};
