import { motion } from "framer-motion";
import { Bot, History, MessageSquareText } from "lucide-react";
import { Link } from "react-router-dom";

import { PageHeader } from "@/components/layout/PageHeader";
import { fadeInUp, staggerChildren } from "@/components/motion/variants";
import { Card, CardContent } from "@/components/ui/Card";
import { TOOL_META } from "@/features/ai-career-coach/pages/CoachToolPage";
import { paths } from "@/routes/paths";

export function CoachHomePage() {
  return (
    <div>
      <PageHeader eyebrow="AI Career Coach" title="Your coach" description="Chat freely, or run a focused tool grounded in your resume and the market." />

      <div className="mb-6 grid gap-3 sm:grid-cols-2">
        <Link to={paths.coachChat}>
          <Card className="flex items-center gap-3 p-5 transition-all hover:-translate-y-0.5 hover:border-brand-border hover:shadow-sm">
            <div className="flex size-11 items-center justify-center rounded-[var(--radius-md)] bg-[linear-gradient(135deg,var(--color-accent-a),var(--color-accent-b))] text-white">
              <MessageSquareText className="size-5" aria-hidden />
            </div>
            <div>
              <p className="text-sm font-semibold text-ink">Open chat</p>
              <p className="text-xs text-ink-secondary">Ask anything, conversationally</p>
            </div>
          </Card>
        </Link>
        <Link to={paths.coachInsights}>
          <Card className="flex items-center gap-3 p-5 transition-all hover:-translate-y-0.5 hover:border-brand-border hover:shadow-sm">
            <div className="flex size-11 items-center justify-center rounded-[var(--radius-md)] bg-surface-2 text-ink-muted">
              <History className="size-5" aria-hidden />
            </div>
            <div>
              <p className="text-sm font-semibold text-ink">Insight history</p>
              <p className="text-xs text-ink-secondary">Revisit past tool results</p>
            </div>
          </Card>
        </Link>
      </div>

      <h2 className="mb-3 flex items-center gap-1.5 text-sm font-semibold text-ink">
        <Bot className="size-4 text-brand" aria-hidden />
        Tools
      </h2>
      <motion.div variants={staggerChildren(0.04)} initial="hidden" animate="visible" className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {Object.entries(TOOL_META).map(([id, meta]) => (
          <motion.div key={id} variants={fadeInUp}>
            <Link to={paths.coachTool(id)}>
              <Card className="h-full p-4 transition-all hover:-translate-y-0.5 hover:border-brand-border hover:shadow-sm">
                <CardContent className="flex flex-col gap-2 p-0">
                  <div className="flex size-9 items-center justify-center rounded-[var(--radius-md)] bg-brand-soft text-brand">
                    <meta.icon className="size-4" aria-hidden />
                  </div>
                  <p className="text-sm font-medium text-ink">{meta.label}</p>
                  <p className="text-xs text-ink-secondary">{meta.description}</p>
                </CardContent>
              </Card>
            </Link>
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
}
