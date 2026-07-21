import { motion } from "framer-motion";
import { Building2, Search, Sparkles } from "lucide-react";

import { Badge } from "@/components/ui/Badge";
import { ScoreRing } from "@/components/ui/ScoreRing";

const MOCK_RESULTS = [
  { title: "Senior Frontend Engineer", company: "Series B, remote-friendly", score: 94 },
  { title: "Staff Product Engineer", company: "Seed stage, hybrid", score: 88 },
  { title: "Frontend Platform Lead", company: "Series A, remote", score: 81 },
];

/** Illustrative product mockup for the landing hero - static design, not live data. */
export function HeroPreview() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24, rotateX: 4 }}
      animate={{ opacity: 1, y: 0, rotateX: 0 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1], delay: 0.15 }}
      className="relative mx-auto w-full max-w-xl rounded-[var(--radius-2xl)] border border-border bg-surface p-2 shadow-lg"
    >
      <div className="rounded-[var(--radius-xl)] border border-border bg-bg p-4">
        <div className="flex items-center gap-2 rounded-[var(--radius-md)] border border-border bg-surface px-3 py-2.5 text-sm text-ink-secondary">
          <Search className="size-4 text-brand" aria-hidden />
          <span>Find me senior frontend roles at Series B startups</span>
          <Sparkles className="ml-auto size-4 text-brand" aria-hidden />
        </div>

        <div className="mt-3 space-y-2">
          {MOCK_RESULTS.map((result, i) => (
            <motion.div
              key={result.title}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4, delay: 0.4 + i * 0.12 }}
              className="flex items-center gap-3 rounded-[var(--radius-md)] border border-border bg-surface p-3"
            >
              <div className="flex size-8 items-center justify-center rounded-[var(--radius-sm)] bg-surface-2 text-ink-muted">
                <Building2 className="size-4" aria-hidden />
              </div>
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-ink">{result.title}</p>
                <p className="truncate text-xs text-ink-muted">{result.company}</p>
              </div>
              <ScoreRing value={result.score} size={36} strokeWidth={4} />
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.4, delay: 0.9 }}
          className="mt-3 flex items-center gap-2"
        >
          <Badge tone="brand">
            <Sparkles className="size-3" aria-hidden /> AI matched on 12 shared skills
          </Badge>
        </motion.div>
      </div>
    </motion.div>
  );
}
