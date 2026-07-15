import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { useAuth } from "@/features/auth";
import {
  activateATSResumeVersion,
  createATSResume,
  createATSResumeVersion,
  deleteATSResume,
  deleteATSResumeVersion,
  getATSResumeDownloadUrl,
  getATSResumeVersion,
  getATSResumeVersionDiff,
  listATSResumes,
  listATSResumeVersions,
} from "@/features/ats-resume-generator/api";
import type { ATSFileFormat, ATSVersionStatus } from "@/features/ats-resume-generator/types";

const POLL_STATUSES: ATSVersionStatus[] = ["pending", "optimizing", "generating_files"];

function scoreLabel(value: number | null): string {
  return value === null ? "—" : value.toFixed(1);
}

function statusLabel(status: ATSVersionStatus): string {
  return status.replace(/_/g, " ");
}

/**
 * ATS Resume Generator feature — real UI (create/optimize, version history,
 * diff view, scores, DOCX/PDF downloads).
 *
 * `resume_parser` has no resume-browsing UI of its own yet, so — mirroring
 * `features/matching/pages/MatchingPage.tsx` — this page takes the
 * *already-parsed* source resume version id as a plain text input
 * (persisted to localStorage), rather than inventing a resume picker that
 * doesn't exist anywhere else in this frontend yet.
 */
export function ATSResumeGeneratorPage() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  const [sourceResumeVersionId, setSourceResumeVersionId] = useState(
    () => localStorage.getItem("ats:sourceResumeVersionId") ?? "",
  );
  const [jobDescription, setJobDescription] = useState("");
  const [title, setTitle] = useState("");
  const [jobTitle, setJobTitle] = useState("");
  const [companyName, setCompanyName] = useState("");
  const [selectedAtsResumeId, setSelectedAtsResumeId] = useState<string | null>(null);
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [compareTo, setCompareTo] = useState("source");

  const effectiveUserId = user?.id ?? null;

  const atsResumesQuery = useQuery({
    queryKey: ["ats-resume-generator", "list", effectiveUserId],
    queryFn: () => listATSResumes(effectiveUserId as string),
    enabled: !!effectiveUserId,
  });

  const versionsQuery = useQuery({
    queryKey: ["ats-resume-generator", "versions", selectedAtsResumeId],
    queryFn: () => listATSResumeVersions(effectiveUserId as string, selectedAtsResumeId as string),
    enabled: !!effectiveUserId && !!selectedAtsResumeId,
  });

  const versionDetailQuery = useQuery({
    queryKey: ["ats-resume-generator", "version", selectedAtsResumeId, selectedVersionId],
    queryFn: () =>
      getATSResumeVersion(
        effectiveUserId as string,
        selectedAtsResumeId as string,
        selectedVersionId as string,
      ),
    enabled: !!effectiveUserId && !!selectedAtsResumeId && !!selectedVersionId,
    refetchInterval: (query) =>
      query.state.data && POLL_STATUSES.includes(query.state.data.status) ? 3000 : false,
  });

  const diffQuery = useQuery({
    queryKey: ["ats-resume-generator", "diff", selectedAtsResumeId, selectedVersionId, compareTo],
    queryFn: () =>
      getATSResumeVersionDiff(
        effectiveUserId as string,
        selectedAtsResumeId as string,
        selectedVersionId as string,
        compareTo,
      ),
    enabled: !!effectiveUserId && !!selectedAtsResumeId && !!selectedVersionId,
  });

  const createResume = useMutation({
    mutationFn: () =>
      createATSResume(effectiveUserId as string, {
        resume_version_id: sourceResumeVersionId,
        job_description: jobDescription,
        title: title || null,
        job_title: jobTitle || null,
        company_name: companyName || null,
      }),
    onSuccess: (resume) => {
      queryClient.invalidateQueries({ queryKey: ["ats-resume-generator", "list"] });
      setSelectedAtsResumeId(resume.id);
      setSelectedVersionId(resume.current_version?.id ?? null);
    },
  });

  const createVersion = useMutation({
    mutationFn: () =>
      createATSResumeVersion(effectiveUserId as string, selectedAtsResumeId as string, {}),
    onSuccess: (version) => {
      queryClient.invalidateQueries({
        queryKey: ["ats-resume-generator", "versions", selectedAtsResumeId],
      });
      setSelectedVersionId(version.id);
    },
  });

  const activateVersion = useMutation({
    mutationFn: (versionId: string) =>
      activateATSResumeVersion(effectiveUserId as string, selectedAtsResumeId as string, versionId),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["ats-resume-generator", "versions", selectedAtsResumeId],
      });
      queryClient.invalidateQueries({ queryKey: ["ats-resume-generator", "list"] });
    },
  });

  const deleteVersion = useMutation({
    mutationFn: (versionId: string) =>
      deleteATSResumeVersion(effectiveUserId as string, selectedAtsResumeId as string, versionId),
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: ["ats-resume-generator", "versions", selectedAtsResumeId],
      }),
  });

  const deleteResume = useMutation({
    mutationFn: () => deleteATSResume(effectiveUserId as string, selectedAtsResumeId as string),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["ats-resume-generator", "list"] });
      setSelectedAtsResumeId(null);
      setSelectedVersionId(null);
    },
  });

  async function handleDownload(format: ATSFileFormat) {
    if (!effectiveUserId || !selectedAtsResumeId || !selectedVersionId) return;
    const url = await getATSResumeDownloadUrl(
      effectiveUserId,
      selectedAtsResumeId,
      selectedVersionId,
      format,
    );
    window.open(url, "_blank", "noopener,noreferrer");
  }

  const version = versionDetailQuery.data;

  return (
    <section>
      <h1>ATS Resume Generator</h1>

      <fieldset>
        <legend>Identifiers</legend>
        <label>
          Source resume version ID (already parsed){" "}
          <input
            value={sourceResumeVersionId}
            onChange={(e) => {
              setSourceResumeVersionId(e.target.value);
              localStorage.setItem("ats:sourceResumeVersionId", e.target.value);
            }}
            placeholder="uuid"
          />
        </label>
      </fieldset>

      {effectiveUserId && (
        <>
          <fieldset>
            <legend>Optimize a resume for a job</legend>
            <div>
              <label>
                Title{" "}
                <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="optional" />
              </label>{" "}
              <label>
                Job title{" "}
                <input value={jobTitle} onChange={(e) => setJobTitle(e.target.value)} placeholder="optional" />
              </label>{" "}
              <label>
                Company{" "}
                <input
                  value={companyName}
                  onChange={(e) => setCompanyName(e.target.value)}
                  placeholder="optional"
                />
              </label>
            </div>
            <div>
              <textarea
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
                placeholder="Paste the target job description here…"
                rows={6}
                cols={80}
              />
            </div>
            <button
              onClick={() => createResume.mutate()}
              disabled={
                !sourceResumeVersionId || jobDescription.trim().length < 20 || createResume.isPending
              }
            >
              Optimize resume
            </button>
            {createResume.isSuccess && <span> Queued — Claude is optimizing your resume.</span>}
            {createResume.isError && <span> Failed to create: {String(createResume.error)}</span>}
          </fieldset>

          <fieldset>
            <legend>Your ATS resumes</legend>
            {atsResumesQuery.isLoading && <p>Loading…</p>}
            <ul>
              {atsResumesQuery.data?.data.map((resume) => (
                <li key={resume.id}>
                  <button
                    onClick={() => {
                      setSelectedAtsResumeId(resume.id);
                      setSelectedVersionId(resume.current_version?.id ?? null);
                    }}
                    disabled={resume.id === selectedAtsResumeId}
                  >
                    {resume.title}
                  </button>{" "}
                  {resume.current_version && (
                    <span>
                      v{resume.current_version.version_number} · {statusLabel(resume.current_version.status)} ·
                      ATS score {scoreLabel(resume.current_version.ats_score)}
                    </span>
                  )}
                </li>
              ))}
              {atsResumesQuery.data?.data.length === 0 && <li>No ATS resumes yet.</li>}
            </ul>
          </fieldset>

          {selectedAtsResumeId && (
            <>
              <fieldset>
                <legend>Version history</legend>
                <button onClick={() => createVersion.mutate()} disabled={createVersion.isPending}>
                  Generate new version
                </button>{" "}
                <button onClick={() => deleteResume.mutate()} disabled={deleteResume.isPending}>
                  Delete this ATS resume
                </button>
                <ul>
                  {versionsQuery.data?.map((v) => (
                    <li key={v.id}>
                      <button
                        onClick={() => setSelectedVersionId(v.id)}
                        disabled={v.id === selectedVersionId}
                      >
                        v{v.version_number}
                      </button>{" "}
                      {v.is_current && <strong>(current)</strong>} · {statusLabel(v.status)} · keyword{" "}
                      {scoreLabel(v.keyword_score)} · ATS {scoreLabel(v.ats_score)}{" "}
                      {!v.is_current && (
                        <button onClick={() => activateVersion.mutate(v.id)}>Activate</button>
                      )}{" "}
                      <button onClick={() => deleteVersion.mutate(v.id)}>Delete</button>
                      {v.error_message && <div>Error: {v.error_message}</div>}
                    </li>
                  ))}
                </ul>
              </fieldset>

              {version && (
                <fieldset>
                  <legend>
                    Version {version.version_number} — {statusLabel(version.status)}
                  </legend>

                  <div>
                    <strong>Resume Score</strong> — Keyword score: {scoreLabel(version.keyword_score)} ·
                    ATS score: {scoreLabel(version.ats_score)}
                  </div>

                  <div>
                    <button
                      onClick={() => handleDownload("docx")}
                      disabled={!version.available_formats.includes("docx")}
                    >
                      Download DOCX
                    </button>{" "}
                    <button
                      onClick={() => handleDownload("pdf")}
                      disabled={!version.available_formats.includes("pdf")}
                    >
                      Download PDF
                    </button>
                  </div>

                  <h3>Optimized summary</h3>
                  <p>{version.optimized_summary || "(pending optimization)"}</p>

                  <h3>Optimized skills</h3>
                  <p>{version.optimized_skills.join(", ") || "(pending optimization)"}</p>

                  <h3>Keywords</h3>
                  <div>Matched: {version.matched_keywords.join(", ") || "none"}</div>
                  <div>Missing: {version.missing_keywords.join(", ") || "none"}</div>

                  <h3>Diff view</h3>
                  <label>
                    Compare against{" "}
                    <select value={compareTo} onChange={(e) => setCompareTo(e.target.value)}>
                      <option value="source">original resume (source)</option>
                      {versionsQuery.data
                        ?.filter((v) => v.id !== selectedVersionId)
                        .map((v) => (
                          <option key={v.id} value={v.id}>
                            v{v.version_number}
                          </option>
                        ))}
                    </select>
                  </label>
                  {diffQuery.data && (
                    <div>
                      <p>
                        Summary:{" "}
                        {diffQuery.data.summary_diff.map((segment, i) => (
                          <span
                            key={i}
                            style={{
                              textDecoration: segment.op === "delete" ? "line-through" : undefined,
                              fontWeight: segment.op === "insert" ? "bold" : undefined,
                              opacity: segment.op === "equal" ? 0.8 : 1,
                            }}
                          >
                            {segment.text}{" "}
                          </span>
                        ))}
                      </p>
                      <p>
                        Skills added: {diffQuery.data.skills_diff.added.join(", ") || "none"} · removed:{" "}
                        {diffQuery.data.skills_diff.removed.join(", ") || "none"}
                      </p>
                      <ul>
                        {diffQuery.data.projects_diff.map((project) => (
                          <li key={project.name}>
                            {project.name} — {project.status}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </fieldset>
              )}
            </>
          )}
        </>
      )}
    </section>
  );
}
