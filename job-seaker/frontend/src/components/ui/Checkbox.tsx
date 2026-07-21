import * as RadixCheckbox from "@radix-ui/react-checkbox";
import { Check } from "lucide-react";

import { cn } from "@/lib/cn";

interface CheckboxProps {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  id?: string;
  disabled?: boolean;
  className?: string;
}

export function Checkbox({ checked, onCheckedChange, id, disabled, className }: CheckboxProps) {
  return (
    <RadixCheckbox.Root
      id={id}
      checked={checked}
      onCheckedChange={(value) => onCheckedChange(value === true)}
      disabled={disabled}
      className={cn(
        "flex size-4 items-center justify-center rounded-[4px] border border-border-strong bg-surface",
        "transition-colors duration-150 outline-none",
        "focus-visible:ring-4 focus-visible:ring-brand-soft",
        "data-[state=checked]:border-brand data-[state=checked]:bg-brand",
        "disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
    >
      <RadixCheckbox.Indicator>
        <Check className="size-3 text-brand-ink" strokeWidth={3} aria-hidden />
      </RadixCheckbox.Indicator>
    </RadixCheckbox.Root>
  );
}
