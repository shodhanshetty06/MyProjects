import { motion } from "framer-motion";
import { ArrowLeft, Compass } from "lucide-react";
import { Link } from "react-router-dom";

import { fadeInUp } from "@/components/motion/variants";
import { Button } from "@/components/ui/Button";
import { paths } from "@/routes/paths";

export function NotFoundPage() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-5 bg-bg px-6 text-center">
      <motion.div variants={fadeInUp} initial="hidden" animate="visible" className="space-y-5">
        <div className="mx-auto flex size-14 items-center justify-center rounded-full bg-brand-soft text-brand">
          <Compass className="size-6" aria-hidden />
        </div>
        <div className="space-y-1">
          <p className="text-sm font-medium text-brand">404</p>
          <h1 className="text-2xl font-semibold text-ink">This page drifted off course</h1>
          <p className="mx-auto max-w-sm text-sm text-ink-secondary">
            The page you're looking for doesn't exist or may have moved.
          </p>
        </div>
        <Link to={paths.home}>
          <Button>
            <ArrowLeft className="size-4" aria-hidden />
            Back to home
          </Button>
        </Link>
      </motion.div>
    </div>
  );
}
