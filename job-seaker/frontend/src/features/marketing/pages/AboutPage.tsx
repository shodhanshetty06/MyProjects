import { motion } from "framer-motion";
import { Compass, Heart, ShieldCheck, Sparkles } from "lucide-react";

import { fadeInUp, staggerChildren } from "@/components/motion/variants";
import { GradientBlob } from "@/features/marketing/components/GradientBlob";
import { Section } from "@/features/marketing/components/Section";

const VALUES = [
  {
    icon: Sparkles,
    title: "AI that's grounded, not gimmicky",
    description: "Every AI feature runs against your real resume and application data - not a generic prompt with your name filled in.",
  },
  {
    icon: ShieldCheck,
    title: "Honest about what's real",
    description: "If something isn't wired up yet, we say so - no fake progress bars or manufactured social proof.",
  },
  {
    icon: Compass,
    title: "One workspace, not ten tabs",
    description: "Search, resume, applications, and coaching share the same context so you're not re-entering the same information everywhere.",
  },
  {
    icon: Heart,
    title: "Built for the job search, not around it",
    description: "The job search is stressful enough - the product should reduce busywork, not add another dashboard to check.",
  },
];

export function AboutPage() {
  return (
    <div>
      <div className="relative overflow-hidden">
        <GradientBlob className="-top-24 left-1/2 h-72 w-[40rem] -translate-x-1/2" />
        <Section eyebrow="About" title="We think the job search should feel like one product" className="pb-8">
          <motion.p
            variants={fadeInUp}
            initial="hidden"
            animate="visible"
            className="mx-auto max-w-2xl text-center text-base text-ink-secondary"
          >
            Most job searches happen across a dozen disconnected tools - a job board, a resume editor, a
            spreadsheet, an inbox. We built this as one AI-native workspace instead: search, tailor your
            resume, apply, and track outcomes, all sharing the same context about who you are and what
            you're looking for.
          </motion.p>
        </Section>
      </div>

      <Section eyebrow="What we care about" title="Principles behind the product">
        <motion.div
          variants={staggerChildren(0.08)}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-80px" }}
          className="grid gap-5 sm:grid-cols-2"
        >
          {VALUES.map((value) => (
            <motion.div
              key={value.title}
              variants={fadeInUp}
              className="rounded-[var(--radius-xl)] border border-border bg-surface p-6"
            >
              <div className="mb-3 flex size-9 items-center justify-center rounded-[var(--radius-md)] bg-brand-soft text-brand">
                <value.icon className="size-4" aria-hidden />
              </div>
              <h3 className="text-sm font-semibold text-ink">{value.title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-ink-secondary">{value.description}</p>
            </motion.div>
          ))}
        </motion.div>
      </Section>
    </div>
  );
}
