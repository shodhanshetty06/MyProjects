import type { ReactNode } from "react";

import { EmptyState } from "@/components/ui/EmptyState";
import { cn } from "@/lib/cn";

export interface TableColumn<T> {
  key: string;
  header: string;
  align?: "left" | "right" | "center";
  render: (row: T) => ReactNode;
  className?: string;
}

interface TableProps<T> {
  columns: TableColumn<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  emptyTitle?: string;
  emptyDescription?: string;
  onRowClick?: (row: T) => void;
  className?: string;
}

const ALIGN_CLASSES = { left: "text-left", right: "text-right", center: "text-center" };

export function Table<T>({
  columns,
  rows,
  rowKey,
  emptyTitle = "Nothing here yet",
  emptyDescription,
  onRowClick,
  className,
}: TableProps<T>) {
  if (rows.length === 0) {
    return <EmptyState title={emptyTitle} description={emptyDescription} />;
  }

  return (
    <div className={cn("overflow-x-auto rounded-[var(--radius-lg)] border border-border", className)}>
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-border bg-surface-2">
            {columns.map((col) => (
              <th
                key={col.key}
                className={cn(
                  "whitespace-nowrap px-4 py-2.5 text-xs font-medium uppercase tracking-wide text-ink-muted",
                  ALIGN_CLASSES[col.align ?? "left"],
                )}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={rowKey(row)}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
              className={cn(
                "border-b border-border last:border-0",
                onRowClick && "cursor-pointer transition-colors hover:bg-surface-2",
              )}
            >
              {columns.map((col) => (
                <td
                  key={col.key}
                  className={cn("px-4 py-3 text-ink", ALIGN_CLASSES[col.align ?? "left"], col.className)}
                >
                  {col.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
