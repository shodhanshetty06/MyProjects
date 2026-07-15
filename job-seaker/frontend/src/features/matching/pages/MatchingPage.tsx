import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { useAuth } from "@/features/auth";
import {
  computeBatchMatches,
  dismissRecommendation,
  getPreferences,
  listRecommendations,
  listResumeMatches,
  refreshRecommendations,
  upsertPreferences,
} from "@/features/matching/api";
import type {
  EmploymentType,
  MatchPreferenceInput,
  RecommendationType,
  WorkMode,
} from "@/features/matching/types";
import { useMatchingSocket } from "@/features/matching/useMatchingSocket";

const WORK_MODES: WorkMode[] = ["remote", "hybrid", "onsite"];
const EMPLOYMENT_TYPES: EmploymentType[] = ["full_time", "part_time", "contract", "internship"];
const RECOMMENDATION_TABS: RecommendationType[] = ["job", "company", "skill"];

const EMPTY_PREFERENCES: MatchPreferenceInput = {
  preferred_countries: [],
  preferred_work_modes: [],
  preferred_employment_types: [],
  years_of_experience: null,
  expected_salary_min: null,
  expected_salary_max: null,
  salary_currency: null,
  is_active: true,
};

function toggle<T>(list: T[], value: T): T[] {
  return list.includes(value) ? list.filter((v) => v !== value) : [...list, value];
}

/**
 * Matching feature — real UI (preferences, matches, recommendations,
 * realtime updates).
 */
export function MatchingPage() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  const [resumeVersionId, setResumeVersionId] = useState(
    () => localStorage.getItem("matching:resumeVersionId") ?? "",
  );
  const [preferencesForm, setPreferencesForm] = useState<MatchPreferenceInput>(EMPTY_PREFERENCES);
  const [activeTab, setActiveTab] = useState<RecommendationType>("job");

  const effectiveUserId = user?.id ?? null;

  const socketStatus = useMatchingSocket(effectiveUserId, (event) => {
    if (event.type === "matches_computed" && resumeVersionId) {
      queryClient.invalidateQueries({ queryKey: ["matching", "matches", resumeVersionId] });
    }
    if (event.type === "recommendations_updated") {
      queryClient.invalidateQueries({ queryKey: ["matching", "recommendations"] });
    }
  });

  const preferencesQuery = useQuery({
    queryKey: ["matching", "preferences", effectiveUserId],
    queryFn: () => getPreferences(effectiveUserId as string),
    enabled: !!effectiveUserId,
    retry: false,
  });

  const savePreferences = useMutation({
    mutationFn: () => upsertPreferences(effectiveUserId as string, preferencesForm),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["matching", "preferences"] }),
  });

  const matchesQuery = useQuery({
    queryKey: ["matching", "matches", resumeVersionId],
    queryFn: () => listResumeMatches(effectiveUserId as string, resumeVersionId),
    enabled: !!effectiveUserId && !!resumeVersionId,
  });

  const computeMatches = useMutation({
    mutationFn: () => computeBatchMatches(effectiveUserId as string, resumeVersionId),
  });

  const recommendationsQuery = useQuery({
    queryKey: ["matching", "recommendations", activeTab, effectiveUserId],
    queryFn: () => listRecommendations(effectiveUserId as string, activeTab),
    enabled: !!effectiveUserId,
  });

  const dismiss = useMutation({
    mutationFn: (recommendationId: string) =>
      dismissRecommendation(effectiveUserId as string, recommendationId),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["matching", "recommendations", activeTab] }),
  });

  const refresh = useMutation({
    mutationFn: () => refreshRecommendations(effectiveUserId as string),
  });

  return (
    <section>
      <h1>Matching</h1>

      <fieldset>
        <legend>Identifiers</legend>
        <label>
          Resume version ID{" "}
          <input
            value={resumeVersionId}
            onChange={(e) => {
              setResumeVersionId(e.target.value);
              localStorage.setItem("matching:resumeVersionId", e.target.value);
            }}
            placeholder="uuid"
          />
        </label>{" "}
        <span>Realtime: {socketStatus}</span>
      </fieldset>

      {effectiveUserId && (
        <>
          <fieldset>
            <legend>Preferences</legend>
            {preferencesQuery.data && (
              <p>
                Saved: {preferencesQuery.data.preferred_countries.join(", ") || "no countries"} ·{" "}
                {preferencesQuery.data.preferred_work_modes.join(", ") || "no work modes"}
              </p>
            )}
            <label>
              Preferred countries (comma-separated){" "}
              <input
                value={preferencesForm.preferred_countries.join(", ")}
                onChange={(e) =>
                  setPreferencesForm((prev) => ({
                    ...prev,
                    preferred_countries: e.target.value
                      .split(",")
                      .map((v) => v.trim())
                      .filter(Boolean),
                  }))
                }
              />
            </label>
            <div>
              Work modes:{" "}
              {WORK_MODES.map((mode) => (
                <label key={mode}>
                  <input
                    type="checkbox"
                    checked={preferencesForm.preferred_work_modes.includes(mode)}
                    onChange={() =>
                      setPreferencesForm((prev) => ({
                        ...prev,
                        preferred_work_modes: toggle(prev.preferred_work_modes, mode),
                      }))
                    }
                  />
                  {mode}
                </label>
              ))}
            </div>
            <div>
              Employment types:{" "}
              {EMPLOYMENT_TYPES.map((type) => (
                <label key={type}>
                  <input
                    type="checkbox"
                    checked={preferencesForm.preferred_employment_types.includes(type)}
                    onChange={() =>
                      setPreferencesForm((prev) => ({
                        ...prev,
                        preferred_employment_types: toggle(prev.preferred_employment_types, type),
                      }))
                    }
                  />
                  {type}
                </label>
              ))}
            </div>
            <label>
              Years of experience{" "}
              <input
                type="number"
                min={0}
                value={preferencesForm.years_of_experience ?? ""}
                onChange={(e) =>
                  setPreferencesForm((prev) => ({
                    ...prev,
                    years_of_experience: e.target.value ? Number(e.target.value) : null,
                  }))
                }
              />
            </label>
            <button onClick={() => savePreferences.mutate()} disabled={savePreferences.isPending}>
              Save preferences
            </button>
          </fieldset>

          <fieldset>
            <legend>Matches for this resume</legend>
            <button
              onClick={() => computeMatches.mutate()}
              disabled={!resumeVersionId || computeMatches.isPending}
            >
              Compute matches
            </button>
            {computeMatches.isSuccess && <span> Queued — results appear via realtime update.</span>}
            {matchesQuery.isLoading && <p>Loading matches…</p>}
            <ul>
              {matchesQuery.data?.data.map((match) => (
                <li key={match.id}>
                  <strong>{match.job?.title ?? match.job_id}</strong>
                  {match.job && ` at ${match.job.company_name}`} — score {match.match_score ?? "n/a"}
                  <div>Matched: {match.matched_skills.join(", ") || "none"}</div>
                  <div>Missing: {match.missing_skills.join(", ") || "none"}</div>
                  <ul>
                    {match.strengths.map((s) => (
                      <li key={s}>+ {s}</li>
                    ))}
                    {match.weaknesses.map((w) => (
                      <li key={w}>- {w}</li>
                    ))}
                  </ul>
                </li>
              ))}
            </ul>
          </fieldset>

          <fieldset>
            <legend>Recommendations</legend>
            <button onClick={() => refresh.mutate()} disabled={refresh.isPending}>
              Refresh recommendations
            </button>
            <div>
              {RECOMMENDATION_TABS.map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  disabled={tab === activeTab}
                >
                  {tab}s
                </button>
              ))}
            </div>
            {recommendationsQuery.isLoading && <p>Loading recommendations…</p>}
            <ul>
              {recommendationsQuery.data?.map((rec) => (
                <li key={rec.id}>
                  {rec.display_name} — score {rec.score} — {rec.reason}{" "}
                  <button onClick={() => dismiss.mutate(rec.id)} disabled={dismiss.isPending}>
                    Dismiss
                  </button>
                </li>
              ))}
              {recommendationsQuery.data?.length === 0 && <li>No recommendations yet.</li>}
            </ul>
          </fieldset>
        </>
      )}
    </section>
  );
}
