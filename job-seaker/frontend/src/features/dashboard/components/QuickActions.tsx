import { motion } from "framer-motion";
import { Bot, FileUp, Search, Send } from "lucide-react";
import { Link } from "react-router-dom";

import { fadeInUp, staggerChildren } from "@/components/motion/variants";
import { paths } from "@/routes/paths";

const ACTIONS = [
  { to: paths.search, icon: Search, label: "Search jobs", description: "Semantic, keyword, or hybrid" },
  { to: paths.resumes, icon: FileUp, label: "Upload resume", description: "Parse & version it" },
  { to: paths.autoApply, icon: Send, label: "Auto apply", description: "Queue an application" },
  { to: paths.coachChat, icon: Bot, label: "Ask AI coach", description: "Get unstuck fast" },
];

export function QuickActions() {
  return (
    <motion.div
      variants={staggerChildren(0.05)}
      initial="hidden"
      animate="visible"
      className="grid grid-cols-2 gap-3 sm:grid-cols-4"
    >
      {ACTIONS.map((action) => (
        <motion.div key={action.to} variants={fadeInUp}>
          <Link
            to={action.to}
            className="group flex flex-col gap-2 rounded-[var(--radius-lg)] border border-border bg-surface p-4 transition-all hover:-translate-y-0.5 hover:border-brand-border hover:shadow-sm"
          >
            <div className="flex size-8 items-center justify-center rounded-[var(--radius-sm)] bg-brand-soft text-brand transition-colors group-hover:bg-brand group-hover:text-brand-ink">
              <action.icon className="size-4" aria-hidden />
            </div>
            <div>
              <p className="text-sm font-medium text-ink">{action.label}</p>
              <p className="text-xs text-ink-muted">{action.description}</p>
            </div>
          </Link>
        </motion.div>
      ))}
    </motion.div>
  );
}
