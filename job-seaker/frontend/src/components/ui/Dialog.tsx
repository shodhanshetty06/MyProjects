import * as RadixDialog from "@radix-ui/react-dialog";
import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/cn";

interface DialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description?: string;
  children?: ReactNode;
  footer?: ReactNode;
  className?: string;
}

export function Dialog({ open, onOpenChange, title, description, children, footer, className }: DialogProps) {
  return (
    <RadixDialog.Root open={open} onOpenChange={onOpenChange}>
      <AnimatePresence>
        {open && (
          <RadixDialog.Portal forceMount>
            <RadixDialog.Overlay asChild forceMount>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.15 }}
                className="fixed inset-0 z-50 bg-overlay backdrop-blur-[2px]"
              />
            </RadixDialog.Overlay>
            <RadixDialog.Content asChild forceMount>
              <motion.div
                initial={{ opacity: 0, scale: 0.96, y: 8 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.96, y: 8 }}
                transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1] }}
                className={cn(
                  "fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2",
                  "rounded-[var(--radius-xl)] border border-border bg-surface p-6 shadow-lg outline-none",
                  className,
                )}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="space-y-1">
                    <RadixDialog.Title className="text-base font-semibold text-ink">{title}</RadixDialog.Title>
                    {description && (
                      <RadixDialog.Description className="text-sm text-ink-secondary">
                        {description}
                      </RadixDialog.Description>
                    )}
                  </div>
                  <RadixDialog.Close className="rounded-[var(--radius-sm)] p-1 text-ink-muted transition-colors hover:bg-surface-2 hover:text-ink">
                    <X className="size-4" aria-hidden />
                    <span className="sr-only">Close</span>
                  </RadixDialog.Close>
                </div>
                {children && <div className="mt-4">{children}</div>}
                {footer && <div className="mt-6 flex items-center justify-end gap-2">{footer}</div>}
              </motion.div>
            </RadixDialog.Content>
          </RadixDialog.Portal>
        )}
      </AnimatePresence>
    </RadixDialog.Root>
  );
}
