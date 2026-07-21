import { motion } from "framer-motion";
import {
  BarChart3,
  Bot,
  Briefcase,
  Check,
  FileText,
  Search,
} from "lucide-react";
import { useState } from "react";

import { staggerChildren } from "@/components/motion/variants";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/Tabs";
import { GradientBlob } from "@/features/marketing/components/GradientBlob";
import { Section } from "@/features/marketing/components/Section";

interface Category {
  id: string;
  label: string;
  icon: typeof Search;
  title: string;
  description: string;
  points: string[];
}

const CATEGORIES: Category[] = [
  {
    id: "search",
    label: "Search & Discovery",
    icon: Search,
    title: "Search that understands roles, not just keywords",
    description: "Keyword, semantic, and hybrid search modes, backed by autocomplete, trending skills, and a full company directory.",
    points: [
      "Keyword, semantic, and hybrid ranking modes with relevance and AI match scoring",
      "Filter by location, work mode, salary range, experience level, and skills",
      "Autocomplete and query suggestions as you type",
      "Trending skills and trending jobs surfaced from live posting data",
      "Company explorer with per-company open roles and hiring intelligence",
      "AI-generated job and company recommendations, personalized to your resume",
    ],
  },
  {
    id: "resume",
    label: "Resume Toolkit",
    icon: FileText,
    title: "From raw resume to ATS-optimized, in minutes",
    description: "Upload once, then generate as many tailored, scored versions as the roles you're chasing.",
    points: [
      "Drag-and-drop upload with live parsing progress into structured skills, experience, and education",
      "Full version history per resume, with an activate/rollback workflow",
      "AI-generated ATS resume versions tailored to a specific job description",
      "Live ATS score and keyword-match score per version",
      "Section-by-section diff between resume versions",
      "Matched vs. missing keyword visualization against the target job",
    ],
  },
  {
    id: "apply",
    label: "Apply & Track",
    icon: Briefcase,
    title: "Every application, in one timeline",
    description: "Whether you apply yourself or let auto-apply handle it, every step lands in the same tracker.",
    points: [
      "Auto-apply queue with per-attempt status, failure reason, and screenshots",
      "Application tracker with status pipeline: applied through offer or rejection",
      "Merged timeline of status changes and your own notes",
      "Calendar view of application activity",
      "Gmail-matched inbound emails linked back to the right application",
      "Notification center with per-channel preferences",
    ],
  },
  {
    id: "coach",
    label: "AI Career Coach",
    icon: Bot,
    title: "A coach with full context on your resume and the market",
    description: "Every tool below runs against your actual resume data and live market signal, not a generic prompt.",
    points: [
      "Conversational chat with saved history across sessions",
      "Skill-gap analysis against a target role",
      "Multi-milestone career roadmap generation",
      "Salary prediction grounded in market data",
      "Personalized learning and certification recommendations",
      "Interview prep questions and resume review scoring",
    ],
  },
  {
    id: "analytics",
    label: "Analytics",
    icon: BarChart3,
    title: "See what's actually working",
    description: "Funnel, performance, and market analytics computed from your real application and resume history.",
    points: [
      "Application, interview, and offer funnels over time",
      "Resume and ATS resume performance, including score trends",
      "Skill demand personalized against skills you already have",
      "Country demand and hiring trend breakdowns",
      "Response-time analytics and an application activity heatmap",
      "CSV and PDF export of your dashboard",
    ],
  },
];

export function FeaturesPage() {
  const [active, setActive] = useState(CATEGORIES[0].id);
  const current = CATEGORIES.find((c) => c.id === active) ?? CATEGORIES[0];

  return (
    <div>
      <div className="relative overflow-hidden">
        <GradientBlob className="-top-24 right-0 h-72 w-[36rem]" />
        <Section
          eyebrow="Product"
          title="Every stage of the search, built as one product"
          description="Not a bolt-on AI feature - the whole platform, from search to offer, is designed around one shared context."
          className="pb-8"
        />
      </div>

      <Section className="pt-0">
        <Tabs value={active} onValueChange={setActive}>
          <TabsList className="mb-8 flex-wrap justify-center gap-1.5 bg-transparent p-0">
            {CATEGORIES.map((category) => (
              <TabsTrigger
                key={category.id}
                value={category.id}
                className="gap-1.5 border border-border bg-surface data-[state=active]:border-brand-border data-[state=active]:bg-brand-soft data-[state=active]:text-brand"
              >
                <category.icon className="size-3.5" aria-hidden />
                {category.label}
              </TabsTrigger>
            ))}
          </TabsList>

          <TabsContent value={current.id} forceMount>
            <motion.div
              key={current.id}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="grid gap-10 rounded-[var(--radius-2xl)] border border-border bg-surface p-8 sm:p-10 lg:grid-cols-[1fr_1.2fr]"
            >
              <div>
                <div className="mb-4 flex size-11 items-center justify-center rounded-[var(--radius-md)] bg-brand-soft text-brand">
                  <current.icon className="size-5" aria-hidden />
                </div>
                <h2 className="text-xl font-semibold text-ink">{current.title}</h2>
                <p className="mt-2 text-sm leading-relaxed text-ink-secondary">{current.description}</p>
              </div>
              <motion.ul variants={staggerChildren(0.05)} initial="hidden" animate="visible" className="space-y-3">
                {current.points.map((point) => (
                  <motion.li
                    key={point}
                    variants={{ hidden: { opacity: 0, x: 8 }, visible: { opacity: 1, x: 0 } }}
                    className="flex items-start gap-2.5 text-sm text-ink-secondary"
                  >
                    <Check className="mt-0.5 size-4 shrink-0 text-brand" aria-hidden />
                    {point}
                  </motion.li>
                ))}
              </motion.ul>
            </motion.div>
          </TabsContent>
        </Tabs>
      </Section>
    </div>
  );
}
