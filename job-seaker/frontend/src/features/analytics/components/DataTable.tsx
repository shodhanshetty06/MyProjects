import type { ReactNode } from "react";

export interface DataTableColumn<T> {
  key: string;
  header: string;
  align?: "left" | "right";
  render: (row: T) => ReactNode;
}

interface DataTableProps<T> {
  columns: DataTableColumn<T>[];
  rows: T[];
  rowKey: (row: T) => string;
}

export function DataTable<T>({ columns, rows, rowKey }: DataTableProps<T>) {
  if (rows.length === 0) {
    return <p className="analytics-empty">No data for this period.</p>;
  }
  return (
    <table className="analytics-table">
      <thead>
        <tr>
          {columns.map((col) => (
            <th key={col.key} className={col.align === "right" ? "analytics-table__cell--right" : ""}>
              {col.header}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={rowKey(row)}>
            {columns.map((col) => (
              <td key={col.key} className={col.align === "right" ? "analytics-table__cell--right" : ""}>
                {col.render(row)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
