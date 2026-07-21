import { forwardRef, type TextareaHTMLAttributes } from "react";

import { cn } from "@/lib/cn";

export interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  invalid?: boolean;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, invalid, ...props }, ref) => (
    <textarea
      ref={ref}
      aria-invalid={invalid}
      className={cn(
        "min-h-24 w-full rounded-[var(--radius-md)] border border-border bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-muted",
        "transition-[border-color,box-shadow] duration-150 outline-none",
        "hover:border-border-strong",
        "focus:border-brand focus:ring-4 focus:ring-brand-soft",
        "disabled:cursor-not-allowed disabled:opacity-50",
        invalid && "border-danger focus:border-danger focus:ring-danger-soft",
        className,
      )}
      {...props}
    />
  ),
);
Textarea.displayName = "Textarea";
