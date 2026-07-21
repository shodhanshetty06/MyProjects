import { motion } from "framer-motion";
import type { ReactNode } from "react";

import { fadeInUp } from "@/components/motion/variants";
import { cn } from "@/lib/cn";

interface SectionProps {
  eyebrow?: string;
  title?: ReactNode;
  description?: string;
  align?: "left" | "center";
  children?: ReactNode;
  className?: string;
  id?: string;
}

export function Section({ eyebrow, title, description, align = "center", children, className, id }: SectionProps) {
  return (
    <section id={id} className={cn("mx-auto max-w-6xl px-5 py-16 sm:px-8 sm:py-24", className)}>
      {(eyebrow || title || description) && (
        <motion.div
          variants={fadeInUp}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-80px" }}
          className={cn("mb-12 max-w-2xl", align === "center" && "mx-auto text-center")}
        >
          {eyebrow && <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-brand">{eyebrow}</p>}
          {title && <h2 className="text-2xl font-semibold tracking-tight text-ink sm:text-3xl">{title}</h2>}
          {description && <p className="mt-3 text-base text-ink-secondary">{description}</p>}
        </motion.div>
      )}
      {children}
    </section>
  );
}
