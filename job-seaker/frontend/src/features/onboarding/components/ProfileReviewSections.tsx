import { Plus } from "lucide-react";
import type { ReactNode } from "react";

import { Button } from "@/components/ui/Button";
import { Checkbox } from "@/components/ui/Checkbox";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { Textarea } from "@/components/ui/Textarea";
import { ReviewBadge } from "@/features/onboarding/components/ReviewBadge";
import { ReviewItemCard } from "@/features/onboarding/components/ReviewItemCard";
import { TagField } from "@/features/onboarding/components/TagField";
import {
  EMPTY_CERTIFICATION,
  EMPTY_EDUCATION,
  EMPTY_EXPERIENCE,
  EMPTY_LANGUAGE,
  EMPTY_PROJECT,
  nextReviewId,
  type ReviewItem,
} from "@/features/onboarding/reviewState";
import type {
  CertificationItem,
  EducationItem,
  ExperienceItem,
  GpaType,
  LanguageItem,
  ProjectItem,
} from "@/features/resume-parser/types";
import { cn } from "@/lib/cn";

const GPA_TYPE_OPTIONS: { value: GpaType; label: string }[] = [
  { value: "gpa", label: "GPA" },
  { value: "cgpa", label: "CGPA" },
  { value: "percentage", label: "Percentage" },
];

function Field({ label, className, children }: { label: string; className?: string; children: ReactNode }) {
  return (
    <div className={cn("space-y-1", className)}>
      <label className="text-xs font-medium text-ink-secondary">{label}</label>
      {children}
    </div>
  );
}

function itemState<T>(item: ReviewItem<T>) {
  if (item.source === "added") return "added" as const;
  return item.accepted ? ("accepted" as const) : ("ai" as const);
}

function updateItem<T>(items: ReviewItem<T>[], id: string, patch: Partial<T>): ReviewItem<T>[] {
  return items.map((item) => (item.id === id ? { ...item, data: { ...item.data, ...patch } } : item));
}

function acceptItem<T>(items: ReviewItem<T>[], id: string): ReviewItem<T>[] {
  return items.map((item) => (item.id === id ? { ...item, accepted: true } : item));
}

function removeItem<T>(items: ReviewItem<T>[], id: string): ReviewItem<T>[] {
  return items.filter((item) => item.id !== id);
}

function addItem<T>(items: ReviewItem<T>[], empty: T): ReviewItem<T>[] {
  return [...items, { id: nextReviewId(), source: "added", accepted: true, data: empty }];
}

interface ReviewSectionProps {
  title: string;
  isEmpty: boolean;
  onAdd: () => void;
  addLabel: string;
  children: ReactNode;
}

function ReviewSection({ title, isEmpty, onAdd, addLabel, children }: ReviewSectionProps) {
  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-ink">{title}</h2>
          {isEmpty && <ReviewBadge state="missing" />}
        </div>
        <Button variant="outline" size="sm" type="button" onClick={onAdd}>
          <Plus className="size-3.5" aria-hidden />
          {addLabel}
        </Button>
      </div>
      {!isEmpty && <div className="space-y-3">{children}</div>}
    </section>
  );
}

interface SectionProps<T> {
  items: ReviewItem<T>[];
  onChange: (items: ReviewItem<T>[]) => void;
}

export function ExperienceSection({ items, onChange }: SectionProps<ExperienceItem>) {
  return (
    <ReviewSection
      title="Experience"
      isEmpty={items.length === 0}
      addLabel="Add role"
      onAdd={() => onChange(addItem(items, EMPTY_EXPERIENCE))}
    >
      {items.map((item) => (
        <ReviewItemCard
          key={item.id}
          state={itemState(item)}
          onAccept={() => onChange(acceptItem(items, item.id))}
          onDelete={() => onChange(removeItem(items, item.id))}
        >
          <div className="grid grid-cols-2 gap-3">
            <Field label="Title">
              <Input
                value={item.data.title}
                onChange={(e) => onChange(updateItem(items, item.id, { title: e.target.value }))}
              />
            </Field>
            <Field label="Company">
              <Input
                value={item.data.company}
                onChange={(e) => onChange(updateItem(items, item.id, { company: e.target.value }))}
              />
            </Field>
            <Field label="Location">
              <Input
                value={item.data.location ?? ""}
                onChange={(e) => onChange(updateItem(items, item.id, { location: e.target.value || null }))}
              />
            </Field>
            <Field label="Dates">
              <div className="flex items-center gap-2">
                <Input
                  placeholder="Start"
                  value={item.data.start_date ?? ""}
                  onChange={(e) =>
                    onChange(updateItem(items, item.id, { start_date: e.target.value || null }))
                  }
                />
                <Input
                  placeholder="End"
                  disabled={item.data.is_current}
                  value={item.data.is_current ? "" : item.data.end_date ?? ""}
                  onChange={(e) => onChange(updateItem(items, item.id, { end_date: e.target.value || null }))}
                />
              </div>
            </Field>
          </div>
          <label className="flex items-center gap-2 text-sm text-ink-secondary">
            <Checkbox
              checked={item.data.is_current}
              onCheckedChange={(checked) =>
                onChange(updateItem(items, item.id, { is_current: checked, end_date: checked ? null : item.data.end_date }))
              }
            />
            I currently work here
          </label>
          <Field label="Summary">
            <Textarea
              value={item.data.description ?? ""}
              onChange={(e) => onChange(updateItem(items, item.id, { description: e.target.value || null }))}
              className="min-h-16"
            />
          </Field>
          <Field label="Highlights (one per line)">
            <Textarea
              value={item.data.highlights.join("\n")}
              onChange={(e) =>
                onChange(
                  updateItem(items, item.id, {
                    highlights: e.target.value.split("\n").map((line) => line.trim()).filter(Boolean),
                  }),
                )
              }
              className="min-h-16"
            />
          </Field>
        </ReviewItemCard>
      ))}
    </ReviewSection>
  );
}

export function EducationSection({ items, onChange }: SectionProps<EducationItem>) {
  return (
    <ReviewSection
      title="Education"
      isEmpty={items.length === 0}
      addLabel="Add education"
      onAdd={() => onChange(addItem(items, EMPTY_EDUCATION))}
    >
      {items.map((item) => (
        <ReviewItemCard
          key={item.id}
          state={itemState(item)}
          onAccept={() => onChange(acceptItem(items, item.id))}
          onDelete={() => onChange(removeItem(items, item.id))}
        >
          <div className="grid grid-cols-2 gap-3">
            <Field label="Institution">
              <Input
                value={item.data.institution}
                onChange={(e) => onChange(updateItem(items, item.id, { institution: e.target.value }))}
              />
            </Field>
            <Field label="Degree">
              <Input
                value={item.data.degree ?? ""}
                onChange={(e) => onChange(updateItem(items, item.id, { degree: e.target.value || null }))}
              />
            </Field>
            <Field label="Field of study">
              <Input
                value={item.data.field_of_study ?? ""}
                onChange={(e) =>
                  onChange(updateItem(items, item.id, { field_of_study: e.target.value || null }))
                }
              />
            </Field>
            <Field label="Grade">
              <div className="flex gap-2">
                <Select
                  value={item.data.gpa_type ?? ""}
                  onValueChange={(value) =>
                    onChange(updateItem(items, item.id, { gpa_type: (value || null) as GpaType | null }))
                  }
                  options={GPA_TYPE_OPTIONS}
                  placeholder="Type"
                  className="w-32 shrink-0"
                />
                <Input
                  value={item.data.gpa ?? ""}
                  onChange={(e) => onChange(updateItem(items, item.id, { gpa: e.target.value || null }))}
                  placeholder="e.g. 8.5"
                  className="flex-1"
                />
              </div>
            </Field>
            <Field label="Start">
              <Input
                value={item.data.start_date ?? ""}
                onChange={(e) => onChange(updateItem(items, item.id, { start_date: e.target.value || null }))}
              />
            </Field>
            <Field label="End">
              <Input
                value={item.data.end_date ?? ""}
                onChange={(e) => onChange(updateItem(items, item.id, { end_date: e.target.value || null }))}
              />
            </Field>
          </div>
        </ReviewItemCard>
      ))}
    </ReviewSection>
  );
}

export function ProjectsSection({ items, onChange }: SectionProps<ProjectItem>) {
  return (
    <ReviewSection
      title="Projects"
      isEmpty={items.length === 0}
      addLabel="Add project"
      onAdd={() => onChange(addItem(items, EMPTY_PROJECT))}
    >
      {items.map((item) => (
        <ReviewItemCard
          key={item.id}
          state={itemState(item)}
          onAccept={() => onChange(acceptItem(items, item.id))}
          onDelete={() => onChange(removeItem(items, item.id))}
        >
          <div className="grid grid-cols-2 gap-3">
            <Field label="Name">
              <Input
                value={item.data.name}
                onChange={(e) => onChange(updateItem(items, item.id, { name: e.target.value }))}
              />
            </Field>
            <Field label="URL">
              <Input
                value={item.data.url ?? ""}
                onChange={(e) => onChange(updateItem(items, item.id, { url: e.target.value || null }))}
              />
            </Field>
          </div>
          <Field label="Description">
            <Textarea
              value={item.data.description ?? ""}
              onChange={(e) => onChange(updateItem(items, item.id, { description: e.target.value || null }))}
              className="min-h-16"
            />
          </Field>
          <Field label="Technologies">
            <TagField
              values={item.data.technologies}
              onChange={(technologies) => onChange(updateItem(items, item.id, { technologies }))}
              placeholder="e.g. React, PostgreSQL"
            />
          </Field>
        </ReviewItemCard>
      ))}
    </ReviewSection>
  );
}

export function CertificationsSection({ items, onChange }: SectionProps<CertificationItem>) {
  return (
    <ReviewSection
      title="Certifications"
      isEmpty={items.length === 0}
      addLabel="Add certification"
      onAdd={() => onChange(addItem(items, EMPTY_CERTIFICATION))}
    >
      {items.map((item) => (
        <ReviewItemCard
          key={item.id}
          state={itemState(item)}
          onAccept={() => onChange(acceptItem(items, item.id))}
          onDelete={() => onChange(removeItem(items, item.id))}
        >
          <div className="grid grid-cols-2 gap-3">
            <Field label="Name">
              <Input
                value={item.data.name}
                onChange={(e) => onChange(updateItem(items, item.id, { name: e.target.value }))}
              />
            </Field>
            <Field label="Issuer">
              <Input
                value={item.data.issuer ?? ""}
                onChange={(e) => onChange(updateItem(items, item.id, { issuer: e.target.value || null }))}
              />
            </Field>
            <Field label="Issued">
              <Input
                value={item.data.issue_date ?? ""}
                onChange={(e) => onChange(updateItem(items, item.id, { issue_date: e.target.value || null }))}
              />
            </Field>
            <Field label="Expires">
              <Input
                value={item.data.expiry_date ?? ""}
                onChange={(e) => onChange(updateItem(items, item.id, { expiry_date: e.target.value || null }))}
              />
            </Field>
            <Field label="Credential ID" className="col-span-2">
              <Input
                value={item.data.credential_id ?? ""}
                onChange={(e) =>
                  onChange(updateItem(items, item.id, { credential_id: e.target.value || null }))
                }
              />
            </Field>
          </div>
        </ReviewItemCard>
      ))}
    </ReviewSection>
  );
}

export function LanguagesSection({ items, onChange }: SectionProps<LanguageItem>) {
  return (
    <ReviewSection
      title="Languages"
      isEmpty={items.length === 0}
      addLabel="Add language"
      onAdd={() => onChange(addItem(items, EMPTY_LANGUAGE))}
    >
      {items.map((item) => (
        <ReviewItemCard
          key={item.id}
          state={itemState(item)}
          onAccept={() => onChange(acceptItem(items, item.id))}
          onDelete={() => onChange(removeItem(items, item.id))}
        >
          <div className="grid grid-cols-2 gap-3">
            <Field label="Language">
              <Input
                value={item.data.language}
                onChange={(e) => onChange(updateItem(items, item.id, { language: e.target.value }))}
              />
            </Field>
            <Field label="Proficiency">
              <Input
                value={item.data.proficiency ?? ""}
                onChange={(e) =>
                  onChange(updateItem(items, item.id, { proficiency: e.target.value || null }))
                }
                placeholder="e.g. fluent"
              />
            </Field>
          </div>
        </ReviewItemCard>
      ))}
    </ReviewSection>
  );
}
