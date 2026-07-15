import { FormEvent, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { useAuth } from "@/features/auth";
import {
  analyzeCountryDemand,
  analyzeMarketTrends,
  analyzeSkillGap,
  computeJobReadiness,
  deleteConversation,
  generateRoadmap,
  getConversation,
  listConversations,
  listInsights,
  predictSalary,
  prepareInterview,
  recommendLearning,
  reviewResume,
  sendChatMessage,
  suggestCareers,
} from "@/features/ai-career-coach/api";
import type {
  CareerInsightType,
  CareerRoadmapResult,
  CareerSuggestionResult,
  CoachConversationRead,
  CountryDemandResult,
  InterviewPrepResult,
  JobReadinessScoreResult,
  LearningRecommendationResult,
  MarketTrendResult,
  ResumeReviewResult,
  SalaryPredictionResult,
  SkillGapResult,
} from "@/features/ai-career-coach/types";

type ToolKey =
  | "review"
  | "career_suggestions"
  | "skill_gap"
  | "learning"
  | "salary"
  | "roadmap"
  | "interview"
  | "readiness"
  | "market"
  | "country";

const TOOLS: { key: ToolKey; label: string; needsResume: boolean; needsRole: boolean }[] = [
  { key: "review", label: "Resume Review", needsResume: true, needsRole: false },
  { key: "career_suggestions", label: "Career Suggestions", needsResume: true, needsRole: false },
  { key: "skill_gap", label: "Skill Gap Analysis", needsResume: true, needsRole: true },
  { key: "learning", label: "Learning Recommendations", needsResume: true, needsRole: true },
  { key: "salary", label: "Salary Prediction", needsResume: false, needsRole: true },
  { key: "roadmap", label: "Career Roadmap", needsResume: true, needsRole: true },
  { key: "interview", label: "Interview Preparation", needsResume: true, needsRole: true },
  { key: "readiness", label: "Job Readiness Score", needsResume: true, needsRole: false },
  { key: "market", label: "Market Trend Analysis", needsResume: false, needsRole: false },
  { key: "country", label: "Country Based Demand", needsResume: false, needsRole: false },
];

function usePersistedState(key: string, initial = "") {
  const [value, setValue] = useState(() => localStorage.getItem(key) ?? initial);
  const update = (next: string) => {
    setValue(next);
    localStorage.setItem(key, next);
  };
  return [value, update] as const;
}

function ScoreBar({ label, value }: { label: string; value: number }) {
  return (
    <div>
      {label}: <strong>{value.toFixed(1)}</strong>/100
    </div>
  );
}

function ResumeReviewView({ result }: { result: ResumeReviewResult }) {
  return (
    <div>
      <ScoreBar label="Overall score" value={result.overall_score} />
      <p>{result.summary}</p>
      <p>
        <strong>Strengths:</strong> {result.strengths.join("; ") || "none"}
      </p>
      <p>
        <strong>Weaknesses:</strong> {result.weaknesses.join("; ") || "none"}
      </p>
      <ul>
        {result.improvement_suggestions.map((s, i) => (
          <li key={i}>
            [{s.priority}] {s.section}: {s.issue} — {s.suggestion}
          </li>
        ))}
      </ul>
    </div>
  );
}

function CareerSuggestionsView({ result }: { result: CareerSuggestionResult }) {
  return (
    <div>
      <p>{result.summary}</p>
      <ul>
        {result.suggested_roles.map((role) => (
          <li key={role.title}>
            <strong>{role.title}</strong> — fit {role.fit_score.toFixed(0)}/100 — {role.match_reason}
          </li>
        ))}
      </ul>
    </div>
  );
}

function SkillGapView({ result }: { result: SkillGapResult }) {
  return (
    <div>
      <ScoreBar label="Readiness for this role" value={result.overall_readiness_percent} />
      <p>
        <strong>Already have:</strong> {result.matched_skills.join(", ") || "none"}
      </p>
      <ul>
        {result.missing_skills.map((item) => (
          <li key={item.skill}>
            [{item.importance}] {item.skill} — {item.reason}
          </li>
        ))}
      </ul>
    </div>
  );
}

function LearningView({ result }: { result: LearningRecommendationResult }) {
  return (
    <ul>
      {result.recommendations.map((rec, i) => (
        <li key={i}>
          [{rec.priority}] {rec.skill}: {rec.title} ({rec.resource_type}
          {rec.provider ? `, ${rec.provider}` : ""}) — {rec.description}
        </li>
      ))}
      {result.recommendations.length === 0 && <li>No gaps found — nothing to recommend.</li>}
    </ul>
  );
}

function SalaryView({ result }: { result: SalaryPredictionResult }) {
  return (
    <div>
      <p>
        <strong>
          {result.predicted_min.toLocaleString()} – {result.predicted_max.toLocaleString()}{" "}
          {result.currency}
        </strong>{" "}
        {result.based_on_market_data
          ? `(based on ${result.sample_size} real job postings)`
          : "(AI estimate — insufficient market data)"}
      </p>
      <p>{result.narrative}</p>
    </div>
  );
}

function RoadmapView({ result }: { result: CareerRoadmapResult }) {
  return (
    <div>
      {result.milestones.map((m, i) => (
        <div key={i}>
          <h4>{m.phase}</h4>
          <p>
            <strong>Goals:</strong> {m.goals.join("; ")}
          </p>
          <p>
            <strong>Skills to develop:</strong> {m.skills_to_develop.join(", ")}
          </p>
          <p>
            <strong>Actions:</strong> {m.suggested_actions.join("; ")}
          </p>
        </div>
      ))}
    </div>
  );
}

function InterviewView({ result }: { result: InterviewPrepResult }) {
  return (
    <div>
      <ul>
        {result.questions.map((q, i) => (
          <li key={i}>
            [{q.category}] {q.question}
            <div>Tip: {q.tip}</div>
          </li>
        ))}
      </ul>
      <p>
        <strong>General tips:</strong> {result.general_tips.join("; ")}
      </p>
    </div>
  );
}

function ReadinessView({ result }: { result: JobReadinessScoreResult }) {
  return (
    <div>
      <ScoreBar label="Overall readiness" value={result.overall_score} />
      <ScoreBar label="Resume quality" value={result.resume_quality_score} />
      <ScoreBar label="Skill match" value={result.skill_match_score} />
      <ScoreBar label="Profile completeness" value={result.completeness_score} />
      <p>{result.narrative}</p>
      <ul>
        {result.recommendations.map((r, i) => (
          <li key={i}>{r}</li>
        ))}
      </ul>
    </div>
  );
}

function MarketTrendView({ result }: { result: MarketTrendResult }) {
  return (
    <div>
      <p>
        {result.total_postings} postings in the last {result.window_days} days (previous window:{" "}
        {result.previous_window_postings}
        {result.growth_percent !== null ? `, growth ${result.growth_percent}%` : ""})
      </p>
      <p>
        <strong>Top skills:</strong>{" "}
        {result.top_skills.map((s) => `${s.skill} (${s.job_count})`).join(", ") || "none"}
      </p>
      <p>
        <strong>Top companies:</strong>{" "}
        {result.top_companies.map((c) => `${c.company} (${c.job_count})`).join(", ") || "none"}
      </p>
      <p>{result.narrative}</p>
    </div>
  );
}

function CountryDemandView({ result }: { result: CountryDemandResult }) {
  return (
    <div>
      <ul>
        {result.countries.map((c) => (
          <li key={c.country}>
            {c.country} — {c.job_count} postings
            {c.avg_salary_min !== null && c.avg_salary_max !== null
              ? ` — avg ${c.avg_salary_min.toFixed(0)}-${c.avg_salary_max.toFixed(0)} ${c.currency ?? ""}`
              : ""}
          </li>
        ))}
        {result.countries.length === 0 && <li>No matching postings.</li>}
      </ul>
      <p>{result.narrative}</p>
    </div>
  );
}

function InsightsPanel({ userId, resumeVersionId }: { userId: string; resumeVersionId: string }) {
  const [activeTool, setActiveTool] = useState<ToolKey>("review");
  const [targetRole, setTargetRole] = usePersistedState("career-coach:targetRole", "");
  const [country, setCountry] = usePersistedState("career-coach:country", "");
  const [yearsExperience, setYearsExperience] = useState<string>("");
  const [timeframeMonths, setTimeframeMonths] = useState(12);

  const tool = TOOLS.find((t) => t.key === activeTool)!;

  const run = useMutation({
    mutationFn: async () => {
      switch (activeTool) {
        case "review":
          return { kind: "review" as const, data: await reviewResume(userId, resumeVersionId) };
        case "career_suggestions":
          return {
            kind: "career_suggestions" as const,
            data: await suggestCareers(userId, resumeVersionId),
          };
        case "skill_gap":
          return {
            kind: "skill_gap" as const,
            data: await analyzeSkillGap(userId, resumeVersionId, targetRole),
          };
        case "learning":
          return {
            kind: "learning" as const,
            data: await recommendLearning(userId, resumeVersionId, targetRole),
          };
        case "salary":
          return {
            kind: "salary" as const,
            data: await predictSalary(
              userId,
              targetRole,
              country || null,
              yearsExperience ? Number(yearsExperience) : null,
              resumeVersionId || null,
            ),
          };
        case "roadmap":
          return {
            kind: "roadmap" as const,
            data: await generateRoadmap(userId, resumeVersionId, targetRole, timeframeMonths),
          };
        case "interview":
          return {
            kind: "interview" as const,
            data: await prepareInterview(userId, resumeVersionId, targetRole),
          };
        case "readiness":
          return {
            kind: "readiness" as const,
            data: await computeJobReadiness(userId, resumeVersionId, targetRole || null),
          };
        case "market":
          return {
            kind: "market" as const,
            data: await analyzeMarketTrends(userId, targetRole || null),
          };
        case "country":
          return {
            kind: "country" as const,
            data: await analyzeCountryDemand(userId, targetRole || null, null),
          };
      }
    },
  });

  const canRun =
    (!tool.needsResume || !!resumeVersionId) && (!tool.needsRole || !!targetRole.trim());

  return (
    <div>
      <nav>
        {TOOLS.map((t) => (
          <button
            key={t.key}
            onClick={() => {
              setActiveTool(t.key);
              run.reset();
            }}
            disabled={t.key === activeTool}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <fieldset>
        <legend>{tool.label} inputs</legend>
        {tool.needsRole && (
          <label>
            Target role{" "}
            <input
              value={targetRole}
              onChange={(e) => setTargetRole(e.target.value)}
              placeholder="e.g. Backend Engineer"
            />
          </label>
        )}
        {activeTool === "salary" && (
          <>
            {" "}
            <label>
              Country{" "}
              <input value={country} onChange={(e) => setCountry(e.target.value)} placeholder="optional" />
            </label>{" "}
            <label>
              Years experience{" "}
              <input
                type="number"
                min={0}
                value={yearsExperience}
                onChange={(e) => setYearsExperience(e.target.value)}
              />
            </label>
          </>
        )}
        {activeTool === "roadmap" && (
          <label>
            {" "}
            Timeframe (months){" "}
            <input
              type="number"
              min={1}
              max={60}
              value={timeframeMonths}
              onChange={(e) => setTimeframeMonths(Number(e.target.value))}
            />
          </label>
        )}
        {tool.needsResume && !resumeVersionId && <p>Set a Resume Version ID above to use this tool.</p>}
        <div>
          <button onClick={() => run.mutate()} disabled={!canRun || run.isPending}>
            {run.isPending ? "Generating…" : `Run ${tool.label}`}
          </button>
        </div>
      </fieldset>

      {run.isError && <p>Something went wrong generating this insight. Try again.</p>}

      {run.data?.kind === "review" && <ResumeReviewView result={run.data.data} />}
      {run.data?.kind === "career_suggestions" && <CareerSuggestionsView result={run.data.data} />}
      {run.data?.kind === "skill_gap" && <SkillGapView result={run.data.data} />}
      {run.data?.kind === "learning" && <LearningView result={run.data.data} />}
      {run.data?.kind === "salary" && <SalaryView result={run.data.data} />}
      {run.data?.kind === "roadmap" && <RoadmapView result={run.data.data} />}
      {run.data?.kind === "interview" && <InterviewView result={run.data.data} />}
      {run.data?.kind === "readiness" && <ReadinessView result={run.data.data} />}
      {run.data?.kind === "market" && <MarketTrendView result={run.data.data} />}
      {run.data?.kind === "country" && <CountryDemandView result={run.data.data} />}
    </div>
  );
}

function ChatPanel({ userId }: { userId: string }) {
  const queryClient = useQueryClient();
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  const conversationsQuery = useQuery({
    queryKey: ["career-coach", "conversations", userId],
    queryFn: () => listConversations(userId),
  });

  const conversationQuery = useQuery({
    queryKey: ["career-coach", "conversation", userId, conversationId],
    queryFn: () => getConversation(userId, conversationId as string),
    enabled: !!conversationId,
  });

  const send = useMutation({
    mutationFn: (message: string) => sendChatMessage(userId, message, conversationId),
    onSuccess: (response) => {
      setConversationId(response.conversation.id);
      setDraft("");
      queryClient.invalidateQueries({ queryKey: ["career-coach", "conversations", userId] });
      queryClient.invalidateQueries({
        queryKey: ["career-coach", "conversation", userId, response.conversation.id],
      });
    },
  });

  const remove = useMutation({
    mutationFn: (id: string) => deleteConversation(userId, id),
    onSuccess: (_data, id) => {
      if (id === conversationId) setConversationId(null);
      queryClient.invalidateQueries({ queryKey: ["career-coach", "conversations", userId] });
    },
  });

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (draft.trim()) send.mutate(draft.trim());
  };

  return (
    <div>
      <aside>
        <h3>Conversations</h3>
        <button onClick={() => setConversationId(null)}>New conversation</button>
        <ul>
          {conversationsQuery.data?.data.map((c: CoachConversationRead) => (
            <li key={c.id}>
              <button onClick={() => setConversationId(c.id)} disabled={c.id === conversationId}>
                {c.title}
              </button>{" "}
              <button onClick={() => remove.mutate(c.id)} disabled={remove.isPending}>
                Delete
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <div>
        <ul>
          {(conversationQuery.data?.messages ?? []).map((m) => (
            <li key={m.id}>
              <strong>{m.role}:</strong> {m.content}
            </li>
          ))}
          {!conversationId && <li>Starting a new conversation — say hello below.</li>}
        </ul>

        <form onSubmit={submit}>
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Ask your career coach anything…"
          />
          <button type="submit" disabled={send.isPending || !draft.trim()}>
            {send.isPending ? "Sending…" : "Send"}
          </button>
        </form>
      </div>
    </div>
  );
}

const INSIGHT_TYPES: CareerInsightType[] = [
  "resume_review",
  "career_suggestions",
  "skill_gap_analysis",
  "learning_recommendations",
  "salary_prediction",
  "career_roadmap",
  "interview_preparation",
  "job_readiness_score",
  "market_trend_analysis",
  "country_demand",
];

function HistoryPanel({ userId }: { userId: string }) {
  const [filter, setFilter] = useState<CareerInsightType | "">("");

  const insightsQuery = useQuery({
    queryKey: ["career-coach", "insights", userId, filter],
    queryFn: () => listInsights(userId, filter || undefined),
  });

  return (
    <div>
      <label>
        Filter by type{" "}
        <select value={filter} onChange={(e) => setFilter(e.target.value as CareerInsightType | "")}>
          <option value="">All</option>
          {INSIGHT_TYPES.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
      </label>
      <ul>
        {insightsQuery.data?.data.map((insight) => (
          <li key={insight.id}>
            [{insight.insight_type}] {new Date(insight.created_at).toLocaleString()}
          </li>
        ))}
        {insightsQuery.data?.data.length === 0 && <li>No insights generated yet.</li>}
      </ul>
    </div>
  );
}

/**
 * AI Career Coach feature — real UI covering all twelve coaching tools plus
 * the chat assistant and insight history.
 */
export function AICareerCoachPage() {
  const { user } = useAuth();
  const [resumeVersionId, setResumeVersionId] = usePersistedState("career-coach:resumeVersionId");
  const [activeTab, setActiveTab] = useState<"insights" | "chat" | "history">("insights");

  const effectiveUserId = user?.id ?? null;

  return (
    <section>
      <h1>AI Career Coach</h1>

      <fieldset>
        <legend>Identifiers</legend>
        <label>
          Resume version ID{" "}
          <input
            value={resumeVersionId}
            onChange={(e) => setResumeVersionId(e.target.value)}
            placeholder="uuid (required for most tools)"
          />
        </label>
      </fieldset>

      {effectiveUserId && (
        <>
          <nav>
            <button onClick={() => setActiveTab("insights")} disabled={activeTab === "insights"}>
              Insights
            </button>{" "}
            <button onClick={() => setActiveTab("chat")} disabled={activeTab === "chat"}>
              Chat Assistant
            </button>{" "}
            <button onClick={() => setActiveTab("history")} disabled={activeTab === "history"}>
              History
            </button>
          </nav>

          {activeTab === "insights" && (
            <InsightsPanel userId={effectiveUserId} resumeVersionId={resumeVersionId.trim()} />
          )}
          {activeTab === "chat" && <ChatPanel userId={effectiveUserId} />}
          {activeTab === "history" && <HistoryPanel userId={effectiveUserId} />}
        </>
      )}
    </section>
  );
}
