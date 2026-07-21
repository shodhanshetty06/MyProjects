import * as RadixTabs from "@radix-ui/react-tabs";
import type { ReactNode } from "react";

import { cn } from "@/lib/cn";

export const Tabs = RadixTabs.Root;

export function TabsList({ className, ...props }: React.ComponentProps<typeof RadixTabs.List>) {
  return (
    <RadixTabs.List
      className={cn("inline-flex items-center gap-1 rounded-[var(--radius-md)] bg-surface-2 p-1", className)}
      {...props}
    />
  );
}

export function TabsTrigger({
  className,
  children,
  ...props
}: React.ComponentProps<typeof RadixTabs.Trigger> & { children: ReactNode }) {
  return (
    <RadixTabs.Trigger
      className={cn(
        "relative rounded-[var(--radius-sm)] px-3 py-1.5 text-sm font-medium text-ink-secondary outline-none transition-colors",
        "hover:text-ink data-[state=active]:bg-surface data-[state=active]:text-ink data-[state=active]:shadow-xs",
        "focus-visible:ring-2 focus-visible:ring-brand-soft",
        className,
      )}
      {...props}
    >
      {children}
    </RadixTabs.Trigger>
  );
}

export function TabsContent({ className, ...props }: React.ComponentProps<typeof RadixTabs.Content>) {
  return <RadixTabs.Content className={cn("outline-none", className)} {...props} />;
}
