import type { ReactNode } from "react";

interface ChartCardProps {
  title: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}

export function ChartCard({ title, subtitle, children, className }: ChartCardProps) {
  return (
    <section className={`analytics-card${className ? ` ${className}` : ""}`}>
      <header className="analytics-card__header">
        <h2 className="analytics-card__title">{title}</h2>
        {subtitle && <p className="analytics-card__subtitle">{subtitle}</p>}
      </header>
      <div className="analytics-card__body">{children}</div>
    </section>
  );
}
