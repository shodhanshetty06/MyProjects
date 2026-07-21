import { GripVertical } from "lucide-react";
import { useState } from "react";
import { Group, Panel, Separator, type Layout } from "react-resizable-panels";

import { cn } from "@/lib/cn";

interface ResizableSplitProps {
  left: React.ReactNode;
  right: React.ReactNode;
  leftDefaultSize?: number;
  leftMinSize?: number;
  rightMinSize?: number;
  /** When set, the panel split is remembered per-browser under this key. */
  persistKey?: string;
  className?: string;
}

function loadLayout(key: string | undefined, fallback: Layout): Layout {
  if (!key) return fallback;
  try {
    const raw = localStorage.getItem(`resizable-split:${key}`);
    return raw ? (JSON.parse(raw) as Layout) : fallback;
  } catch {
    return fallback;
  }
}

/**
 * Thin styling wrapper over react-resizable-panels for the two-pane
 * layouts used by the dashboard and AI coach ("Resizable Panels" in the
 * brief) - keeps the drag-handle affordance and optional persistence
 * consistent everywhere it's used.
 */
export function ResizableSplit({
  left,
  right,
  leftDefaultSize = 68,
  leftMinSize = 40,
  rightMinSize = 22,
  persistKey,
  className,
}: ResizableSplitProps) {
  const [layout, setLayout] = useState<Layout>(() =>
    loadLayout(persistKey, { left: leftDefaultSize, right: 100 - leftDefaultSize }),
  );

  return (
    <Group
      orientation="horizontal"
      defaultLayout={layout}
      onLayoutChanged={(next) => {
        setLayout(next);
        if (persistKey) {
          localStorage.setItem(`resizable-split:${persistKey}`, JSON.stringify(next));
        }
      }}
      className={cn("flex", className)}
    >
      <Panel id="left" defaultSize={leftDefaultSize} minSize={leftMinSize}>
        {left}
      </Panel>
      <Separator className="group relative mx-1.5 flex w-1.5 cursor-col-resize items-center justify-center rounded-full outline-none">
        <div className="h-full w-px bg-border transition-colors group-hover:bg-brand group-data-[resize-handle-active]:bg-brand" />
        <div className="absolute flex size-5 items-center justify-center rounded-full border border-border bg-surface text-ink-muted opacity-0 shadow-xs transition-opacity group-hover:opacity-100">
          <GripVertical className="size-3" aria-hidden />
        </div>
      </Separator>
      <Panel id="right" minSize={rightMinSize}>
        {right}
      </Panel>
    </Group>
  );
}
