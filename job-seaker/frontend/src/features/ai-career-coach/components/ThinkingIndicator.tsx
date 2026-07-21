import { motion } from "framer-motion";
import { Sparkles } from "lucide-react";

export function ThinkingIndicator({ label = "Thinking..." }: { label?: string }) {
  return (
    <div className="flex items-center gap-2.5 text-sm text-ink-secondary">
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ repeat: Infinity, duration: 1.4, ease: "linear" }}
        className="flex size-6 items-center justify-center rounded-full bg-brand-soft text-brand"
      >
        <Sparkles className="size-3.5" aria-hidden />
      </motion.div>
      <span>{label}</span>
      <span className="flex gap-0.5">
        {[0, 1, 2].map((i) => (
          <motion.span
            key={i}
            animate={{ opacity: [0.2, 1, 0.2] }}
            transition={{ repeat: Infinity, duration: 1.2, delay: i * 0.15 }}
            className="size-1 rounded-full bg-ink-muted"
          />
        ))}
      </span>
    </div>
  );
}
