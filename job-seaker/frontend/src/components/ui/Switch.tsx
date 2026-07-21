import * as RadixSwitch from "@radix-ui/react-switch";

import { cn } from "@/lib/cn";

interface SwitchProps {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  id?: string;
  disabled?: boolean;
  "aria-label"?: string;
}

export function Switch({ checked, onCheckedChange, id, disabled, ...props }: SwitchProps) {
  return (
    <RadixSwitch.Root
      id={id}
      checked={checked}
      onCheckedChange={onCheckedChange}
      disabled={disabled}
      className={cn(
        "relative h-5 w-9 shrink-0 rounded-full bg-surface-2 border border-border transition-colors duration-150 outline-none",
        "focus-visible:ring-4 focus-visible:ring-brand-soft",
        "data-[state=checked]:bg-brand data-[state=checked]:border-brand",
        "disabled:cursor-not-allowed disabled:opacity-50",
      )}
      {...props}
    >
      <RadixSwitch.Thumb className="block size-3.5 translate-x-0.5 rounded-full bg-white shadow-sm transition-transform duration-150 data-[state=checked]:translate-x-[18px]" />
    </RadixSwitch.Root>
  );
}
