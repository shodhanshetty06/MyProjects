import { motion } from "framer-motion";
import { Sparkles } from "lucide-react";
import { Suspense } from "react";
import { Link, Outlet } from "react-router-dom";

import { PageLoader } from "@/components/common/LoadingSpinner";
import { fadeInScale } from "@/components/motion/variants";
import { paths } from "@/routes/paths";

export function AuthShell() {
  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-bg px-4 py-10">
      <div
        aria-hidden
        className="pointer-events-none absolute -top-40 left-1/2 h-96 w-[42rem] -translate-x-1/2 rounded-full opacity-25 blur-3xl"
        style={{
          background: "linear-gradient(135deg,var(--color-accent-a),var(--color-accent-b))",
        }}
      />

      <div className="relative w-full max-w-sm">
        <Link to={paths.home} className="mb-8 flex items-center justify-center gap-2">
          <div className="flex size-7 items-center justify-center rounded-[var(--radius-sm)] bg-[linear-gradient(135deg,var(--color-accent-a),var(--color-accent-b))]">
            <Sparkles className="size-4 text-white" aria-hidden />
          </div>
          <span className="text-sm font-semibold text-ink">Ascend AI</span>
        </Link>

        <motion.div
          variants={fadeInScale}
          initial="hidden"
          animate="visible"
          className="rounded-[var(--radius-xl)] border border-border bg-surface p-8 shadow-md"
        >
          <Suspense fallback={<PageLoader />}>
            <Outlet />
          </Suspense>
        </motion.div>
      </div>
    </div>
  );
}
