import { motion } from "framer-motion";
import { CheckCircle2, XCircle } from "lucide-react";

import { fadeInUp, staggerChildren } from "@/components/motion/variants";
import { Badge, type BadgeTone } from "@/components/ui/Badge";
import { Card, CardContent } from "@/components/ui/Card";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { ScoreRing } from "@/components/ui/ScoreRing";
import type {
  CareerRoadmapResult,
  CareerSuggestionResult,
  CountryDemandResult,
  InterviewPrepResult,
  JobReadinessScoreResult,
  LearningRecommendationResult,
  MarketTrendResult,
  ResumeReviewResult,
  SalaryPredictionResult,
  SkillGapResult,
} from "@/features/ai-career-coach/types";

const PRIORITY_TONE: Record<string, BadgeTone> = { high: "danger", medium: "warning", low: "neutral" };

export function ResumeReviewView({ result }: { result: ResumeReviewResult }) {
  return (
    <motion.div variants={staggerChildren(0.06)} initial="hidden" animate="visible" className="space-y-4">
      <motion.div variants={fadeInUp} className="flex items-center gap-4">
        <ScoreRing value={result.overall_score} size={64} />
        <p className="text-sm text-ink-secondary">{result.summary}</p>
      </motion.div>
      <motion.div variants={fadeInUp} className="grid gap-4 sm:grid-cols-2">
        <div>
          <h3 className="mb-2 text-sm font-semibold text-ink">Strengths</h3>
          <ul className="space-y-1.5">
            {result.strengths.map((s, i) => (
              <li key={i} className="flex items-start gap-1.5 text-sm text-ink-secondary">
                <CheckCircle2 className="mt-0.5 size-3.5 shrink-0 text-success" aria-hidden />
                {s}
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h3 className="mb-2 text-sm font-semibold text-ink">Weaknesses</h3>
          <ul className="space-y-1.5">
            {result.weaknesses.map((w, i) => (
              <li key={i} className="flex items-start gap-1.5 text-sm text-ink-secondary">
                <XCircle className="mt-0.5 size-3.5 shrink-0 text-danger" aria-hidden />
                {w}
              </li>
            ))}
          </ul>
        </div>
      </motion.div>
      {result.improvement_suggestions.length > 0 && (
        <motion.div variants={fadeInUp} className="space-y-2">
          <h3 className="text-sm font-semibold text-ink">Suggestions</h3>
          {result.improvement_suggestions.map((s, i) => (
            <Card key={i}>
              <CardContent className="p-3.5">
                <div className="flex items-center gap-2">
                  <Badge tone={PRIORITY_TONE[s.priority]}>{s.priority}</Badge>
                  <span className="text-xs font-medium text-ink-muted">{s.section}</span>
                </div>
                <p className="mt-1.5 text-sm text-ink-secondary">{s.suggestion}</p>
              </CardContent>
            </Card>
          ))}
        </motion.div>
      )}
    </motion.div>
  );
}

export function CareerSuggestionsView({ result }: { result: CareerSuggestionResult }) {
  return (
    <motion.div variants={staggerChildren(0.06)} initial="hidden" animate="visible" className="space-y-4">
      <motion.p variants={fadeInUp} className="text-sm text-ink-secondary">
        {result.summary}
      </motion.p>
      <motion.div variants={staggerChildren(0.05)} className="grid gap-3 sm:grid-cols-2">
        {result.suggested_roles.map((role) => (
          <motion.div key={role.title} variants={fadeInUp}>
            <Card>
              <CardContent className="flex items-center gap-3 p-4">
                <ScoreRing value={role.fit_score} size={44} strokeWidth={4} />
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-ink">{role.title}</p>
                  <p className="text-xs text-ink-secondary">{role.match_reason}</p>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </motion.div>
    </motion.div>
  );
}

export function SkillGapView({ result }: { result: SkillGapResult }) {
  return (
    <motion.div variants={staggerChildren(0.06)} initial="hidden" animate="visible" className="space-y-4">
      <motion.div variants={fadeInUp} className="flex items-center gap-4">
        <ScoreRing value={result.overall_readiness_percent} size={64} />
        <div>
          <p className="text-sm font-medium text-ink">Readiness for {result.target_role}</p>
          <p className="text-xs text-ink-muted">{result.matched_skills.length} matched skills</p>
        </div>
      </motion.div>
      <motion.div variants={fadeInUp}>
        <h3 className="mb-2 text-sm font-semibold text-ink">Matched skills</h3>
        <div className="flex flex-wrap gap-1.5">
          {result.matched_skills.map((s) => (
            <Badge key={s} tone="success">{s}</Badge>
          ))}
        </div>
      </motion.div>
      <motion.div variants={fadeInUp} className="space-y-2">
        <h3 className="text-sm font-semibold text-ink">Missing skills</h3>
        {result.missing_skills.map((item) => (
          <Card key={item.skill}>
            <CardContent className="flex items-start gap-3 p-3.5">
              <Badge tone={PRIORITY_TONE[item.importance]}>{item.importance}</Badge>
              <div>
                <p className="text-sm font-medium text-ink">{item.skill}</p>
                <p className="text-xs text-ink-secondary">{item.reason}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </motion.div>
    </motion.div>
  );
}

export function LearningRecommendationsView({ result }: { result: LearningRecommendationResult }) {
  return (
    <motion.div variants={staggerChildren(0.05)} initial="hidden" animate="visible" className="space-y-2">
      {result.recommendations.map((rec, i) => (
        <motion.div key={i} variants={fadeInUp}>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <Badge tone="brand">{rec.resource_type}</Badge>
                <Badge tone={PRIORITY_TONE[rec.priority]}>{rec.priority}</Badge>
              </div>
              <p className="mt-1.5 text-sm font-medium text-ink">{rec.title}</p>
              {rec.provider && <p className="text-xs text-ink-muted">{rec.provider}</p>}
              <p className="mt-1 text-xs text-ink-secondary">{rec.description}</p>
            </CardContent>
          </Card>
        </motion.div>
      ))}
    </motion.div>
  );
}

export function RoadmapView({ result }: { result: CareerRoadmapResult }) {
  return (
    <motion.div variants={staggerChildren(0.06)} initial="hidden" animate="visible" className="space-y-4">
      <motion.p variants={fadeInUp} className="text-sm text-ink-secondary">
        {result.timeframe_months}-month roadmap to {result.target_role}
      </motion.p>
      {result.milestones.map((milestone, i) => (
        <motion.div key={i} variants={fadeInUp} className="relative border-l-2 border-border pl-4">
          <div className="absolute -left-[5px] top-1 size-2 rounded-full bg-brand" />
          <p className="text-sm font-semibold text-ink">{milestone.phase}</p>
          <ul className="mt-1 list-disc space-y-0.5 pl-4 text-xs text-ink-secondary">
            {milestone.goals.map((g, gi) => (
              <li key={gi}>{g}</li>
            ))}
          </ul>
          {milestone.skills_to_develop.length > 0 && (
            <div className="mt-1.5 flex flex-wrap gap-1">
              {milestone.skills_to_develop.map((s) => (
                <Badge key={s} tone="neutral" className="text-[10px]">{s}</Badge>
              ))}
            </div>
          )}
        </motion.div>
      ))}
    </motion.div>
  );
}

export function SalaryPredictionView({ result }: { result: SalaryPredictionResult }) {
  const fmt = (n: number) => new Intl.NumberFormat("en-US", { style: "currency", currency: result.currency, maximumFractionDigits: 0 }).format(n);
  return (
    <motion.div variants={fadeInUp} initial="hidden" animate="visible" className="space-y-3">
      <p className="text-2xl font-semibold text-ink">
        {fmt(result.predicted_min)} - {fmt(result.predicted_max)}
      </p>
      <p className="text-sm text-ink-secondary">{result.narrative}</p>
      <div className="flex gap-2 text-xs text-ink-muted">
        <Badge tone={result.based_on_market_data ? "success" : "neutral"}>
          {result.based_on_market_data ? "Based on market data" : "Estimated"}
        </Badge>
        <Badge tone="neutral">{result.sample_size} data points</Badge>
      </div>
    </motion.div>
  );
}

export function InterviewPrepView({ result }: { result: InterviewPrepResult }) {
  return (
    <motion.div variants={staggerChildren(0.05)} initial="hidden" animate="visible" className="space-y-3">
      {result.questions.map((q, i) => (
        <motion.div key={i} variants={fadeInUp}>
          <Card>
            <CardContent className="p-4">
              <Badge tone="brand">{q.category}</Badge>
              <p className="mt-1.5 text-sm font-medium text-ink">{q.question}</p>
              <p className="mt-1 text-xs text-ink-secondary">{q.tip}</p>
            </CardContent>
          </Card>
        </motion.div>
      ))}
      {result.general_tips.length > 0 && (
        <motion.div variants={fadeInUp}>
          <h3 className="mb-1.5 text-sm font-semibold text-ink">General tips</h3>
          <ul className="list-disc space-y-1 pl-4 text-sm text-ink-secondary">
            {result.general_tips.map((t, i) => (
              <li key={i}>{t}</li>
            ))}
          </ul>
        </motion.div>
      )}
    </motion.div>
  );
}

export function JobReadinessView({ result }: { result: JobReadinessScoreResult }) {
  return (
    <motion.div variants={staggerChildren(0.06)} initial="hidden" animate="visible" className="space-y-4">
      <motion.div variants={fadeInUp} className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Card className="flex flex-col items-center gap-1.5 p-4">
          <ScoreRing value={result.overall_score} size={52} />
          <p className="text-[11px] text-ink-muted">Overall</p>
        </Card>
        <Card className="flex flex-col items-center gap-1.5 p-4">
          <ScoreRing value={result.resume_quality_score} size={52} />
          <p className="text-[11px] text-ink-muted">Resume quality</p>
        </Card>
        <Card className="flex flex-col items-center gap-1.5 p-4">
          <ScoreRing value={result.skill_match_score} size={52} />
          <p className="text-[11px] text-ink-muted">Skill match</p>
        </Card>
        <Card className="flex flex-col items-center gap-1.5 p-4">
          <ScoreRing value={result.completeness_score} size={52} />
          <p className="text-[11px] text-ink-muted">Completeness</p>
        </Card>
      </motion.div>
      <motion.p variants={fadeInUp} className="text-sm text-ink-secondary">
        {result.narrative}
      </motion.p>
      {result.recommendations.length > 0 && (
        <motion.ul variants={fadeInUp} className="list-disc space-y-1 pl-4 text-sm text-ink-secondary">
          {result.recommendations.map((r, i) => (
            <li key={i}>{r}</li>
          ))}
        </motion.ul>
      )}
    </motion.div>
  );
}

export function MarketTrendsView({ result }: { result: MarketTrendResult }) {
  return (
    <motion.div variants={staggerChildren(0.06)} initial="hidden" animate="visible" className="space-y-4">
      <motion.div variants={fadeInUp} className="flex items-center gap-4">
        <div>
          <p className="text-2xl font-semibold text-ink">{result.total_postings.toLocaleString()}</p>
          <p className="text-xs text-ink-muted">postings in {result.window_days} days</p>
        </div>
        {result.growth_percent != null && (
          <Badge tone={result.growth_percent >= 0 ? "success" : "danger"}>
            {result.growth_percent >= 0 ? "+" : ""}{result.growth_percent.toFixed(1)}%
          </Badge>
        )}
      </motion.div>
      <motion.p variants={fadeInUp} className="text-sm text-ink-secondary">{result.narrative}</motion.p>
      <motion.div variants={fadeInUp} className="grid gap-4 sm:grid-cols-2">
        <div>
          <h3 className="mb-2 text-sm font-semibold text-ink">Top skills</h3>
          <div className="space-y-1.5">
            {result.top_skills.map((s) => (
              <div key={s.skill} className="flex items-center justify-between text-sm">
                <span className="text-ink-secondary">{s.skill}</span>
                <span className="text-xs text-ink-muted">{s.job_count}</span>
              </div>
            ))}
          </div>
        </div>
        <div>
          <h3 className="mb-2 text-sm font-semibold text-ink">Top companies</h3>
          <div className="space-y-1.5">
            {result.top_companies.map((c) => (
              <div key={c.company} className="flex items-center justify-between text-sm">
                <span className="text-ink-secondary">{c.company}</span>
                <span className="text-xs text-ink-muted">{c.job_count}</span>
              </div>
            ))}
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}

export function CountryDemandView({ result }: { result: CountryDemandResult }) {
  const max = Math.max(...result.countries.map((c) => c.job_count), 1);
  return (
    <motion.div variants={staggerChildren(0.05)} initial="hidden" animate="visible" className="space-y-3">
      <motion.p variants={fadeInUp} className="text-sm text-ink-secondary">{result.narrative}</motion.p>
      {result.countries.map((c) => (
        <motion.div key={c.country} variants={fadeInUp} className="space-y-1">
          <div className="flex items-center justify-between text-sm">
            <span className="text-ink">{c.country}</span>
            <span className="text-xs text-ink-muted">{c.job_count} jobs</span>
          </div>
          <ProgressBar value={c.job_count} max={max} />
        </motion.div>
      ))}
    </motion.div>
  );
}
