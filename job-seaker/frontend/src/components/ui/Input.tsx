import { forwardRef, type InputHTMLAttributes, type ReactNode } from "react";

import { cn } from "@/lib/cn";

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  leadingIcon?: ReactNode;
  trailingIcon?: ReactNode;
  invalid?: boolean;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, leadingIcon, trailingIcon, invalid, ...props }, ref) => {
    if (leadingIcon || trailingIcon) {
      return (
        <div className="relative flex items-center">
          {leadingIcon && (
            <span className="pointer-events-none absolute left-3 flex text-ink-muted [&>svg]:size-4">
              {leadingIcon}
            </span>
          )}
          <input
            ref={ref}
            aria-invalid={invalid}
            className={cn(baseInputClasses, leadingIcon && "pl-9", trailingIcon && "pr-9", invalid && invalidClasses, className)}
            {...props}
          />
          {trailingIcon && (
            <span className="absolute right-3 flex text-ink-muted [&>svg]:size-4">{trailingIcon}</span>
          )}
        </div>
      );
    }
    return (
      <input
        ref={ref}
        aria-invalid={invalid}
        className={cn(baseInputClasses, invalid && invalidClasses, className)}
        {...props}
      />
    );
  },
);
Input.displayName = "Input";

const baseInputClasses = cn(
  "h-9 w-full rounded-[var(--radius-md)] border border-border bg-surface px-3 text-sm text-ink placeholder:text-ink-muted",
  "transition-[border-color,box-shadow] duration-150 outline-none",
  "hover:border-border-strong",
  "focus:border-brand focus:ring-4 focus:ring-brand-soft",
  "disabled:cursor-not-allowed disabled:opacity-50",
);

const invalidClasses = "border-danger focus:border-danger focus:ring-danger-soft";
