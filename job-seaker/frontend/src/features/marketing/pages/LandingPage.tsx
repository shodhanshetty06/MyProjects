import { motion } from "framer-motion";
import {
  ArrowRight,
  BarChart3,
  Bot,
  Briefcase,
  FileText,
  Search,
  Send,
  Sparkles,
} from "lucide-react";
import { Link } from "react-router-dom";

import { fadeInUp, staggerChildren } from "@/components/motion/variants";
import { Button } from "@/components/ui/Button";
import { FeatureCard } from "@/features/marketing/components/FeatureCard";
import { GradientBlob } from "@/features/marketing/components/GradientBlob";
import { HeroPreview } from "@/features/marketing/components/HeroPreview";
import { Section } from "@/features/marketing/components/Section";
import { paths } from "@/routes/paths";

const FEATURES = [
  {
    icon: Search,
    title: "Semantic job search",
    description: "Search the way you'd describe a role to a friend - keyword, semantic, or hybrid matching finds roles that fit, not just keywords that match.",
  },
  {
    icon: FileText,
    title: "ATS resume optimization",
    description: "Generate a tailored, keyword-optimized version of your resume for any job description, with a live ATS score and keyword diff.",
  },
  {
    icon: Send,
    title: "Auto apply",
    description: "Queue applications against your saved profile and resume, and track every attempt, screenshot, and outcome in one place.",
  },
  {
    icon: Briefcase,
    title: "Application tracking",
    description: "A single timeline for every application - status changes, notes, and matched inbound emails, from applied to offer.",
  },
  {
    icon: Bot,
    title: "AI career coach",
    description: "Skill-gap analysis, career roadmaps, salary predictions, and interview prep grounded in real market and resume data.",
  },
  {
    icon: BarChart3,
    title: "Career analytics",
    description: "Funnels, response times, and skill demand trends that show what's actually working in your search.",
  },
];

const STEPS = [
  { title: "Upload your resume", description: "We parse it into structured skills, experience, and education in seconds." },
  { title: "Search or let AI find roles", description: "Semantic search and AI recommendations surface roles matched to your profile." },
  { title: "Tailor and apply", description: "Generate an ATS-optimized resume version and apply - manually or automatically." },
  { title: "Track and improve", description: "Watch your funnel, respond to matched emails, and let the coach close your skill gaps." },
];

export function LandingPage() {
  return (
    <div>
      <div className="relative overflow-hidden">
        <GradientBlob className="-top-32 left-1/2 h-96 w-[48rem] -translate-x-1/2" />
        <Section className="pb-8 pt-16 sm:pt-24" align="center">
          <motion.div variants={fadeInUp} initial="hidden" animate="visible" className="mx-auto max-w-2xl text-center">
            <span className="mb-5 inline-flex items-center gap-1.5 rounded-full border border-brand-border bg-brand-soft px-3 py-1 text-xs font-medium text-brand">
              <Sparkles className="size-3.5" aria-hidden />
              AI-native job search
            </span>
            <h1 className="text-4xl font-semibold tracking-tight text-ink sm:text-5xl">
              Your job search, run by an AI that actually gets your career.
            </h1>
            <p className="mx-auto mt-4 max-w-xl text-base text-ink-secondary sm:text-lg">
              Search, tailor, apply, and track - all in one workspace built around a career coach
              that understands your resume, your skill gaps, and the market.
            </p>
            <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
              <Link to={paths.register}>
                <Button size="lg" variant="gradient">
                  Get started free
                  <ArrowRight className="size-4" aria-hidden />
                </Button>
              </Link>
              <Link to={paths.features}>
                <Button size="lg" variant="secondary">
                  See how it works
                </Button>
              </Link>
            </div>
          </motion.div>

          <div className="mt-16">
            <HeroPreview />
          </div>
        </Section>
      </div>

      <Section
        eyebrow="Everything, connected"
        title="One workspace for the entire search"
        description="Every part of the job hunt, wired into the same AI context - so nothing gets rebuilt from scratch at each step."
      >
        <motion.div
          variants={staggerChildren(0.08)}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-80px" }}
          className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3"
        >
          {FEATURES.map((feature) => (
            <FeatureCard key={feature.title} {...feature} />
          ))}
        </motion.div>
      </Section>

      <Section eyebrow="How it works" title="From resume to offer, in four steps" className="bg-surface-2/50">
        <motion.div
          variants={staggerChildren(0.1)}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-80px" }}
          className="grid gap-8 sm:grid-cols-2 lg:grid-cols-4"
        >
          {STEPS.map((step, i) => (
            <motion.div key={step.title} variants={fadeInUp} className="relative">
              <div className="mb-3 flex size-8 items-center justify-center rounded-full bg-ink text-xs font-semibold text-bg">
                {i + 1}
              </div>
              <h3 className="text-sm font-semibold text-ink">{step.title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-ink-secondary">{step.description}</p>
            </motion.div>
          ))}
        </motion.div>
      </Section>

      <Section align="center" className="pb-24">
        <motion.div
          variants={fadeInUp}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="relative overflow-hidden rounded-[var(--radius-2xl)] border border-border bg-ink px-8 py-14 text-center"
        >
          <GradientBlob className="-bottom-24 left-1/2 h-64 w-[36rem] -translate-x-1/2 opacity-30" />
          <div className="relative">
            <h2 className="text-2xl font-semibold text-bg sm:text-3xl">Ready to let AI run your search?</h2>
            <p className="mx-auto mt-3 max-w-md text-sm text-bg/70">
              Create a free account and upload your resume to get matched in minutes.
            </p>
            <Link to={paths.register} className="mt-7 inline-block">
              <Button size="lg" variant="gradient">
                Get started free
                <ArrowRight className="size-4" aria-hidden />
              </Button>
            </Link>
          </div>
        </motion.div>
      </Section>
    </div>
  );
}
