import { AnimatePresence, motion } from "framer-motion";
import { AlertCircle, AlertTriangle, CheckCircle2, Info, X } from "lucide-react";
import { useEffect } from "react";

import { cn } from "@/lib/cn";
import { type Toast, type ToastTone, useToastStore } from "@/stores/toastStore";

const TONE_ICON: Record<ToastTone, typeof Info> = {
  default: Info,
  success: CheckCircle2,
  warning: AlertTriangle,
  danger: AlertCircle,
};

const TONE_ICON_CLASSES: Record<ToastTone, string> = {
  default: "text-brand",
  success: "text-success",
  warning: "text-warning",
  danger: "text-danger",
};

function ToastItem({ toast }: { toast: Toast }) {
  const dismiss = useToastStore((state) => state.dismiss);
  const Icon = TONE_ICON[toast.tone];

  useEffect(() => {
    const timer = window.setTimeout(() => dismiss(toast.id), toast.duration);
    return () => window.clearTimeout(timer);
  }, [toast.id, toast.duration, dismiss]);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: -8, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, x: 40, transition: { duration: 0.15 } }}
      transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
      role="status"
      className="pointer-events-auto flex w-80 items-start gap-3 rounded-[var(--radius-lg)] border border-border bg-surface p-3.5 shadow-lg"
    >
      <Icon className={cn("mt-0.5 size-4 shrink-0", TONE_ICON_CLASSES[toast.tone])} aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-ink">{toast.title}</p>
        {toast.description && <p className="mt-0.5 text-xs text-ink-secondary">{toast.description}</p>}
      </div>
      <button
        type="button"
        onClick={() => dismiss(toast.id)}
        className="shrink-0 text-ink-muted transition-colors hover:text-ink"
      >
        <X className="size-3.5" aria-hidden />
        <span className="sr-only">Dismiss</span>
      </button>
    </motion.div>
  );
}

export function Toaster() {
  const toasts = useToastStore((state) => state.toasts);
  return (
    <div
      aria-live="polite"
      className="pointer-events-none fixed right-4 top-4 z-[100] flex flex-col gap-2"
    >
      <AnimatePresence initial={false}>
        {toasts.map((toast) => (
          <ToastItem key={toast.id} toast={toast} />
        ))}
      </AnimatePresence>
    </div>
  );
}
