import * as RadixSelect from "@radix-ui/react-select";
import { Check, ChevronDown } from "lucide-react";
import { type ReactNode } from "react";

import { cn } from "@/lib/cn";

export interface SelectOption {
  value: string;
  label: string;
}

interface SelectProps {
  value: string;
  onValueChange: (value: string) => void;
  options: SelectOption[];
  placeholder?: string;
  className?: string;
  triggerIcon?: ReactNode;
  disabled?: boolean;
}

export function Select({ value, onValueChange, options, placeholder, className, triggerIcon, disabled }: SelectProps) {
  return (
    <RadixSelect.Root value={value} onValueChange={onValueChange} disabled={disabled}>
      <RadixSelect.Trigger
        className={cn(
          "flex h-9 w-full items-center justify-between gap-2 rounded-[var(--radius-md)] border border-border bg-surface px-3 text-sm text-ink",
          "transition-colors duration-150 outline-none hover:border-border-strong",
          "focus:border-brand focus:ring-4 focus:ring-brand-soft",
          "disabled:cursor-not-allowed disabled:opacity-50 data-[placeholder]:text-ink-muted",
          className,
        )}
      >
        <span className="flex items-center gap-2 truncate">
          {triggerIcon}
          <RadixSelect.Value placeholder={placeholder} />
        </span>
        <RadixSelect.Icon>
          <ChevronDown className="size-4 text-ink-muted" aria-hidden />
        </RadixSelect.Icon>
      </RadixSelect.Trigger>
      <RadixSelect.Portal>
        <RadixSelect.Content
          position="popper"
          sideOffset={6}
          className="z-50 max-h-64 min-w-[var(--radix-select-trigger-width)] overflow-hidden rounded-[var(--radius-md)] border border-border bg-surface shadow-lg animate-[fade-in_0.15s_ease-out]"
        >
          <RadixSelect.Viewport className="p-1">
            {options.map((option) => (
              <RadixSelect.Item
                key={option.value}
                value={option.value}
                className={cn(
                  "relative flex cursor-pointer select-none items-center rounded-[var(--radius-sm)] py-1.5 pl-7 pr-3 text-sm text-ink outline-none",
                  "data-[highlighted]:bg-surface-2 data-[state=checked]:font-medium",
                )}
              >
                <RadixSelect.ItemIndicator className="absolute left-2 flex items-center">
                  <Check className="size-3.5 text-brand" aria-hidden />
                </RadixSelect.ItemIndicator>
                <RadixSelect.ItemText>{option.label}</RadixSelect.ItemText>
              </RadixSelect.Item>
            ))}
          </RadixSelect.Viewport>
        </RadixSelect.Content>
      </RadixSelect.Portal>
    </RadixSelect.Root>
  );
}
