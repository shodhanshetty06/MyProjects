import { motion } from "framer-motion";
import { Building2, Pin, PinOff } from "lucide-react";
import { Link } from "react-router-dom";

import { fadeInUp } from "@/components/motion/variants";
import type { CompanyRead } from "@/features/job-ingestion/types";
import { paths } from "@/routes/paths";
import { usePinsStore } from "@/stores/pinsStore";

export function CompanyCard({ company }: { company: CompanyRead }) {
  const isPinned = usePinsStore((state) => state.isPinned("company", company.id));
  const togglePin = usePinsStore((state) => state.togglePin);

  return (
    <motion.div
      variants={fadeInUp}
      className="group relative rounded-[var(--radius-lg)] border border-border bg-surface p-4 transition-all hover:-translate-y-0.5 hover:border-brand-border hover:shadow-sm"
    >
      <button
        type="button"
        onClick={() => togglePin({ id: company.id, type: "company", label: company.name, meta: company.industry ?? undefined })}
        className="absolute right-3 top-3 flex size-7 items-center justify-center rounded-[var(--radius-sm)] text-ink-muted opacity-0 transition-opacity hover:bg-surface-2 hover:text-ink group-hover:opacity-100 data-[pinned=true]:opacity-100"
        data-pinned={isPinned}
        aria-label={isPinned ? "Unpin company" : "Pin company"}
      >
        {isPinned ? <PinOff className="size-4 text-brand" aria-hidden /> : <Pin className="size-4" aria-hidden />}
      </button>

      <Link to={paths.companyDetail(company.id)} className="flex flex-col gap-3">
        <div className="flex size-11 items-center justify-center overflow-hidden rounded-[var(--radius-md)] bg-surface-2 text-ink-muted">
          {company.logo_url ? (
            <img
              src={company.logo_url}
              alt=""
              width={44}
              height={44}
              loading="lazy"
              decoding="async"
              className="size-full object-cover"
            />
          ) : (
            <Building2 className="size-5" aria-hidden />
          )}
        </div>
        <div className="min-w-0 pr-6">
          <p className="truncate text-sm font-semibold text-ink">{company.name}</p>
          {company.industry && <p className="truncate text-xs text-ink-muted">{company.industry}</p>}
        </div>
        {company.description && (
          <p className="line-clamp-2 text-xs text-ink-secondary">{company.description}</p>
        )}
      </Link>
    </motion.div>
  );
}
