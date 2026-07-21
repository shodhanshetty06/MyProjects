import { motion } from "framer-motion";
import { RotateCcw, ServerCrash } from "lucide-react";

import { fadeInUp } from "@/components/motion/variants";
import { Button } from "@/components/ui/Button";

export function ServerErrorPage() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-5 bg-bg px-6 text-center">
      <motion.div variants={fadeInUp} initial="hidden" animate="visible" className="space-y-5">
        <div className="mx-auto flex size-14 items-center justify-center rounded-full bg-danger-soft text-danger">
          <ServerCrash className="size-6" aria-hidden />
        </div>
        <div className="space-y-1">
          <p className="text-sm font-medium text-danger">500</p>
          <h1 className="text-2xl font-semibold text-ink">Something broke on our end</h1>
          <p className="mx-auto max-w-sm text-sm text-ink-secondary">
            We hit an unexpected error serving this page. Try again in a moment.
          </p>
        </div>
        <Button onClick={() => window.location.reload()}>
          <RotateCcw className="size-4" aria-hidden />
          Reload
        </Button>
      </motion.div>
    </div>
  );
}
