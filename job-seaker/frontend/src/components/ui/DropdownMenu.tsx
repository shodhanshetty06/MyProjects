import * as RadixDropdown from "@radix-ui/react-dropdown-menu";
import type { ReactNode } from "react";

import { cn } from "@/lib/cn";

export const DropdownMenu = RadixDropdown.Root;
export const DropdownMenuTrigger = RadixDropdown.Trigger;

export function DropdownMenuContent({
  children,
  align = "end",
  className,
}: {
  children: ReactNode;
  align?: "start" | "center" | "end";
  className?: string;
}) {
  return (
    <RadixDropdown.Portal>
      <RadixDropdown.Content
        align={align}
        sideOffset={6}
        className={cn(
          "z-50 min-w-[10rem] overflow-hidden rounded-[var(--radius-md)] border border-border bg-surface p-1 shadow-lg",
          "data-[state=open]:animate-[fade-in_0.15s_ease-out]",
          className,
        )}
      >
        {children}
      </RadixDropdown.Content>
    </RadixDropdown.Portal>
  );
}

export function DropdownMenuItem({
  children,
  className,
  destructive,
  ...props
}: React.ComponentProps<typeof RadixDropdown.Item> & { destructive?: boolean }) {
  return (
    <RadixDropdown.Item
      className={cn(
        "flex cursor-pointer select-none items-center gap-2 rounded-[var(--radius-sm)] px-2.5 py-1.5 text-sm outline-none",
        "text-ink data-[highlighted]:bg-surface-2",
        destructive && "text-danger data-[highlighted]:bg-danger-soft",
        className,
      )}
      {...props}
    >
      {children}
    </RadixDropdown.Item>
  );
}

export function DropdownMenuLabel({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <RadixDropdown.Label className={cn("px-2.5 py-1.5 text-xs font-medium text-ink-muted", className)}>
      {children}
    </RadixDropdown.Label>
  );
}

export function DropdownMenuSeparator({ className }: { className?: string }) {
  return <RadixDropdown.Separator className={cn("my-1 h-px bg-border", className)} />;
}
