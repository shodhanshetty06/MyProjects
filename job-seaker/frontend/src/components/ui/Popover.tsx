import * as RadixPopover from "@radix-ui/react-popover";
import type { ReactNode } from "react";

import { cn } from "@/lib/cn";

export const Popover = RadixPopover.Root;
export const PopoverTrigger = RadixPopover.Trigger;
export const PopoverAnchor = RadixPopover.Anchor;

export function PopoverContent({
  children,
  align = "center",
  className,
}: {
  children: ReactNode;
  align?: "start" | "center" | "end";
  className?: string;
}) {
  return (
    <RadixPopover.Portal>
      <RadixPopover.Content
        align={align}
        sideOffset={8}
        className={cn(
          "z-50 rounded-[var(--radius-lg)] border border-border bg-surface p-4 shadow-lg outline-none",
          "data-[state=open]:animate-[fade-in_0.15s_ease-out]",
          className,
        )}
      >
        {children}
      </RadixPopover.Content>
    </RadixPopover.Portal>
  );
}
