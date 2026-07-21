import { motion } from "framer-motion";
import { Check, Sparkles } from "lucide-react";
import { Link } from "react-router-dom";

import { fadeInUp, staggerChildren } from "@/components/motion/variants";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { GradientBlob } from "@/features/marketing/components/GradientBlob";
import { Section } from "@/features/marketing/components/Section";
import { cn } from "@/lib/cn";
import { paths } from "@/routes/paths";

interface Plan {
  name: string;
  price: string;
  period?: string;
  description: string;
  features: string[];
  cta: string;
  highlighted?: boolean;
}

const PLANS: Plan[] = [
  {
    name: "Free",
    price: "$0",
    description: "Everything you need to run a focused search.",
    features: [
      "Semantic job search & filters",
      "Resume upload with parsing",
      "1 active resume with version history",
      "Application tracker & timeline",
      "Core AI coach tools",
    ],
    cta: "Get started free",
  },
  {
    name: "Pro",
    price: "$19",
    period: "/mo",
    description: "For an active search across multiple tracks.",
    features: [
      "Everything in Free",
      "Unlimited ATS resume versions",
      "Auto-apply with attempt tracking",
      "Gmail-matched email tracking",
      "Full analytics dashboard & exports",
      "Priority AI coach responses",
    ],
    cta: "Start free trial",
    highlighted: true,
  },
  {
    name: "Enterprise",
    price: "Custom",
    description: "For career centers, bootcamps, and teams.",
    features: [
      "Everything in Pro",
      "Multi-seat admin portal",
      "Cohort-level analytics",
      "SSO & custom onboarding",
      "Dedicated support",
    ],
    cta: "Talk to us",
  },
];

export function PricingPage() {
  return (
    <div>
      <div className="relative overflow-hidden">
        <GradientBlob className="-top-24 left-1/2 h-72 w-[42rem] -translate-x-1/2" />
        <Section
          eyebrow="Pricing"
          title="Simple pricing that scales with your search"
          description="Start free. Upgrade when auto-apply and full analytics start paying for themselves."
          className="pb-8"
        />
      </div>

      <Section className="pt-0">
        <motion.div
          variants={staggerChildren(0.08)}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-80px" }}
          className="grid gap-6 lg:grid-cols-3"
        >
          {PLANS.map((plan) => (
            <motion.div
              key={plan.name}
              variants={fadeInUp}
              className={cn(
                "relative flex flex-col rounded-[var(--radius-2xl)] border p-7",
                plan.highlighted ? "border-brand-border bg-surface shadow-glow" : "border-border bg-surface",
              )}
            >
              {plan.highlighted && (
                <Badge tone="brand" className="absolute -top-3 left-7">
                  <Sparkles className="size-3" aria-hidden /> Most popular
                </Badge>
              )}
              <h3 className="text-sm font-semibold text-ink">{plan.name}</h3>
              <div className="mt-3 flex items-baseline gap-1">
                <span className="text-3xl font-semibold tracking-tight text-ink">{plan.price}</span>
                {plan.period && <span className="text-sm text-ink-muted">{plan.period}</span>}
              </div>
              <p className="mt-2 text-sm text-ink-secondary">{plan.description}</p>

              <ul className="mt-6 flex-1 space-y-2.5">
                {plan.features.map((feature) => (
                  <li key={feature} className="flex items-start gap-2.5 text-sm text-ink-secondary">
                    <Check className="mt-0.5 size-4 shrink-0 text-brand" aria-hidden />
                    {feature}
                  </li>
                ))}
              </ul>

              <Link to={paths.register} className="mt-7">
                <Button className="w-full" variant={plan.highlighted ? "gradient" : "secondary"}>
                  {plan.cta}
                </Button>
              </Link>
            </motion.div>
          ))}
        </motion.div>
      </Section>
    </div>
  );
}
