import { motion } from "framer-motion";

import { cn } from "@/lib/cn";

interface ScoreRingProps {
  value: number;
  max?: number;
  size?: number;
  strokeWidth?: number;
  label?: string;
  toneFor?: (value: number, max: number) => string;
  className?: string;
}

function defaultTone(value: number, max: number): string {
  const pct = (value / max) * 100;
  if (pct >= 75) return "var(--color-success)";
  if (pct >= 45) return "var(--color-warning)";
  return "var(--color-danger)";
}

export function ScoreRing({
  value,
  max = 100,
  size = 72,
  strokeWidth = 6,
  label,
  toneFor = defaultTone,
  className,
}: ScoreRingProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const pct = Math.max(0, Math.min(1, value / max));
  const color = toneFor(value, max);

  return (
    <div className={cn("relative inline-flex items-center justify-center", className)} style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="var(--color-surface-2)"
          strokeWidth={strokeWidth}
        />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: circumference * (1 - pct) }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-base font-semibold text-ink">{Math.round(value)}</span>
        {label && <span className="text-[10px] text-ink-muted">{label}</span>}
      </div>
    </div>
  );
}
