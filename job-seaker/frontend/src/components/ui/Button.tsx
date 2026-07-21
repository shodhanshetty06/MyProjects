import { motion, type HTMLMotionProps } from "framer-motion";
import { Loader2 } from "lucide-react";
import { forwardRef, type ReactNode } from "react";

import { cn } from "@/lib/cn";

const VARIANT_CLASSES = {
  primary:
    "bg-brand text-brand-ink shadow-sm hover:bg-brand-hover active:bg-brand-hover disabled:hover:bg-brand",
  secondary:
    "bg-surface text-ink border border-border shadow-xs hover:bg-surface-hover active:bg-surface-2",
  ghost: "bg-transparent text-ink-secondary hover:bg-surface-2 hover:text-ink",
  outline: "bg-transparent text-ink border border-border hover:bg-surface-2",
  danger: "bg-danger text-white shadow-sm hover:brightness-110",
  gradient:
    "text-white shadow-glow bg-[linear-gradient(135deg,var(--color-accent-a),var(--color-accent-b))] hover:brightness-110",
} as const;

const SIZE_CLASSES = {
  sm: "h-8 px-3 text-xs gap-1.5 rounded-[var(--radius-sm)]",
  md: "h-9 px-4 text-sm gap-2 rounded-[var(--radius-md)]",
  lg: "h-11 px-5 text-sm gap-2 rounded-[var(--radius-md)]",
  icon: "h-9 w-9 rounded-[var(--radius-md)]",
} as const;

export interface ButtonProps extends Omit<HTMLMotionProps<"button">, "children"> {
  variant?: keyof typeof VARIANT_CLASSES;
  size?: keyof typeof SIZE_CLASSES;
  isLoading?: boolean;
  children?: ReactNode;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    { className, variant = "primary", size = "md", isLoading = false, disabled, children, ...props },
    ref,
  ) => {
    return (
      <motion.button
        ref={ref}
        whileTap={{ scale: 0.97 }}
        transition={{ duration: 0.12 }}
        disabled={disabled || isLoading}
        className={cn(
          "inline-flex select-none items-center justify-center whitespace-nowrap font-medium",
          "transition-colors duration-150 focus-visible:outline-2 focus-visible:outline-brand",
          "disabled:cursor-not-allowed disabled:opacity-50",
          VARIANT_CLASSES[variant],
          SIZE_CLASSES[size],
          className,
        )}
        {...props}
      >
        {isLoading && <Loader2 className="size-4 animate-spin" aria-hidden />}
        {children}
      </motion.button>
    );
  },
);
Button.displayName = "Button";
