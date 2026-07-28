import { Check } from "lucide-react";

import { cn } from "@/lib/cn";
import type { ATSResumeTemplateSummary } from "@/features/ats-resume-generator/types";

/**
 * Fast in-app template preview — a CSS swatch built from the template's own
 * font/color metadata (fetched from `GET /ats-resumes/templates`), not a
 * rendered file. Real fidelity preview is "download and open" — this is
 * just enough to compare templates before committing to a download.
 */
export function TemplatePicker({
  templates,
  value,
  onChange,
  disabled,
}: {
  templates: ATSResumeTemplateSummary[];
  value: string;
  onChange: (templateId: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
      {templates.map((template) => {
        const selected = template.id === value;
        return (
          <button
            key={template.id}
            type="button"
            disabled={disabled}
            onClick={() => onChange(template.id)}
            className={cn(
              "relative rounded-[var(--radius-lg)] border p-3 text-left transition-colors disabled:opacity-60",
              selected ? "border-brand-border bg-brand-soft" : "border-border bg-surface hover:bg-surface-2",
            )}
          >
            {selected && (
              <span className="absolute right-2 top-2 flex size-4 items-center justify-center rounded-full bg-brand text-white">
                <Check className="size-3" aria-hidden />
              </span>
            )}
            <div
              className="mb-1.5 h-10 w-full rounded-[var(--radius-sm)] border"
              style={{
                borderColor: template.accent_color ?? "var(--color-border)",
                background: `linear-gradient(180deg, ${template.heading_color}12, transparent 60%)`,
              }}
            >
              <div
                className="h-2 w-3/5 rounded-sm"
                style={{
                  backgroundColor: template.heading_color,
                  marginTop: 6,
                  marginLeft: 6,
                }}
              />
              {template.accent_color && (
                <div
                  className="mt-1 h-px w-4/5"
                  style={{ backgroundColor: template.accent_color, marginLeft: 6 }}
                />
              )}
            </div>
            <p className="truncate text-sm font-medium text-ink" style={{ fontFamily: template.heading_font }}>
              {template.name}
            </p>
            <p className="mt-0.5 line-clamp-2 text-xs text-ink-muted">{template.description}</p>
          </button>
        );
      })}
    </div>
  );
}
