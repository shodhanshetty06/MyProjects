import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { AlertTriangle, Download, FileStack, Plus, RefreshCw, Sparkles, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { fadeInUp, staggerChildren } from "@/components/motion/variants";
import { PageHeader } from "@/components/layout/PageHeader";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent } from "@/components/ui/Card";
import { Dialog } from "@/components/ui/Dialog";
import { EmptyState } from "@/components/ui/EmptyState";
import { Input } from "@/components/ui/Input";
import { ScoreRing } from "@/components/ui/ScoreRing";
import { Select } from "@/components/ui/Select";
import { SkeletonText } from "@/components/ui/Skeleton";
import { Textarea } from "@/components/ui/Textarea";
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
  listATSResumeTemplates,
  setATSResumeVersionTemplate,
} from "@/features/ats-resume-generator/api";
import { DiffText } from "@/features/ats-resume-generator/components/DiffText";
import { TemplatePicker } from "@/features/ats-resume-generator/components/TemplatePicker";
import type { ATSFileFormat, ATSVersionStatus } from "@/features/ats-resume-generator/types";
import { useAuth } from "@/features/auth";
import { getJobDetail } from "@/features/search/api";
import { listResumes } from "@/features/resume-parser";
import { cn } from "@/lib/cn";
import { errorMessage } from "@/lib/errorMessage";
import { formatEnumLabel } from "@/lib/format";
import { toast } from "@/stores/toastStore";

const POLL_STATUSES: ATSVersionStatus[] = ["pending", "optimizing", "generating_files"];

export function ATSResumeGeneratorPage() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  const userId = user!.id;
  const [searchParams] = useSearchParams();

  const [createOpen, setCreateOpen] = useState(false);
  const [sourceResumeVersionId, setSourceResumeVersionId] = useState("");
  const [jobDescription, setJobDescription] = useState("");
  const [title, setTitle] = useState("");
  const [jobTitle, setJobTitle] = useState("");
  const [companyName, setCompanyName] = useState("");
  const [targetJobId, setTargetJobId] = useState<string | null>(null);
  const [selectedAtsResumeId, setSelectedAtsResumeId] = useState<string | null>(null);
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [compareTo, setCompareTo] = useState("source");
  const [templateId, setTemplateId] = useState("modern_ats");

  // Prefills the "optimize for a job" dialog when arriving from a job's
  // "Generate ATS Resume" quick action (`JobCard`/`JobDetailPage`), which
  // links here with `?target_job_id=...&job_title=...&company_name=...`.
  // Best-effort: fetches the job's description too, via search/api's
  // `getJobDetail`, so the user doesn't have to paste it in manually.
  useEffect(() => {
    const jobId = searchParams.get("target_job_id");
    if (!jobId) return;
    setTargetJobId(jobId);
    setJobTitle(searchParams.get("job_title") ?? "");
    setCompanyName(searchParams.get("company_name") ?? "");
    setCreateOpen(true);
    void getJobDetail(jobId)
      .then((job) => setJobDescription(job.description ?? ""))
      .catch(() => {
        // Best-effort prefill only — user can still paste the description manually.
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const resumesQuery = useQuery({
    queryKey: ["resumes", userId, "picker"],
    queryFn: () => listResumes(userId, 1, 50),
  });
  const parsedResumeOptions = (resumesQuery.data?.data ?? [])
    .filter((r) => r.current_version?.status === "parsed")
    .map((r) => ({ value: r.current_version!.id, label: r.title }));

  const atsResumesQuery = useQuery({
    queryKey: ["ats-resume-generator", "list", userId],
    queryFn: () => listATSResumes(userId),
  });

  const templatesQuery = useQuery({
    queryKey: ["ats-resume-generator", "templates"],
    queryFn: () => listATSResumeTemplates(userId),
    staleTime: Infinity,
  });
  const templates = templatesQuery.data ?? [];

  const versionsQuery = useQuery({
    queryKey: ["ats-resume-generator", "versions", selectedAtsResumeId],
    queryFn: () => listATSResumeVersions(userId, selectedAtsResumeId as string),
    enabled: !!selectedAtsResumeId,
  });

  const versionDetailQuery = useQuery({
    queryKey: ["ats-resume-generator", "version", selectedAtsResumeId, selectedVersionId],
    queryFn: () => getATSResumeVersion(userId, selectedAtsResumeId as string, selectedVersionId as string),
    enabled: !!selectedAtsResumeId && !!selectedVersionId,
    refetchInterval: (query) => (query.state.data && POLL_STATUSES.includes(query.state.data.status) ? 3000 : false),
  });

  const diffQuery = useQuery({
    queryKey: ["ats-resume-generator", "diff", selectedAtsResumeId, selectedVersionId, compareTo],
    queryFn: () => getATSResumeVersionDiff(userId, selectedAtsResumeId as string, selectedVersionId as string, compareTo),
    enabled: !!selectedAtsResumeId && !!selectedVersionId,
  });

  const createResume = useMutation({
    mutationFn: () =>
      createATSResume(userId, {
        resume_version_id: sourceResumeVersionId,
        job_description: jobDescription,
        title: title || null,
        job_title: jobTitle || null,
        company_name: companyName || null,
        target_job_id: targetJobId,
        template_id: templateId,
      }),
    onSuccess: (resume) => {
      toast.success("Optimizing your resume", "This usually takes under a minute.");
      void queryClient.invalidateQueries({ queryKey: ["ats-resume-generator", "list"] });
      setSelectedAtsResumeId(resume.id);
      setSelectedVersionId(resume.current_version?.id ?? null);
      setCreateOpen(false);
      setJobDescription("");
      setTitle("");
      setJobTitle("");
      setCompanyName("");
      setTargetJobId(null);
      setTemplateId("modern_ats");
    },
    onError: (error) =>
      toast.danger("Couldn't start optimization", errorMessage(error, "Check the job description and try again.")),
  });

  const createVersion = useMutation({
    mutationFn: () => createATSResumeVersion(userId, selectedAtsResumeId as string, {}),
    onSuccess: (version) => {
      void queryClient.invalidateQueries({ queryKey: ["ats-resume-generator", "versions", selectedAtsResumeId] });
      setSelectedVersionId(version.id);
    },
  });

  const switchTemplate = useMutation({
    mutationFn: (newTemplateId: string) =>
      setATSResumeVersionTemplate(userId, selectedAtsResumeId as string, selectedVersionId as string, newTemplateId),
    onSuccess: () => {
      toast.success("Switching template", "Re-rendering your DOCX/PDF with the new look.");
      void queryClient.invalidateQueries({ queryKey: ["ats-resume-generator", "version", selectedAtsResumeId, selectedVersionId] });
      void queryClient.invalidateQueries({ queryKey: ["ats-resume-generator", "versions", selectedAtsResumeId] });
    },
    onError: (error) => toast.danger("Couldn't switch template", errorMessage(error)),
  });

  const activateVersion = useMutation({
    mutationFn: (versionId: string) => activateATSResumeVersion(userId, selectedAtsResumeId as string, versionId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["ats-resume-generator", "versions", selectedAtsResumeId] });
      void queryClient.invalidateQueries({ queryKey: ["ats-resume-generator", "list"] });
    },
  });

  const deleteVersion = useMutation({
    mutationFn: (versionId: string) => deleteATSResumeVersion(userId, selectedAtsResumeId as string, versionId),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ["ats-resume-generator", "versions", selectedAtsResumeId] }),
  });

  const deleteResume = useMutation({
    mutationFn: () => deleteATSResume(userId, selectedAtsResumeId as string),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["ats-resume-generator", "list"] });
      setSelectedAtsResumeId(null);
      setSelectedVersionId(null);
    },
  });

  async function handleDownload(format: ATSFileFormat) {
    if (!selectedAtsResumeId || !selectedVersionId) return;
    const url = await getATSResumeDownloadUrl(userId, selectedAtsResumeId, selectedVersionId, format);
    window.open(url, "_blank", "noopener,noreferrer");
  }

  const version = versionDetailQuery.data;
  const atsResumes = atsResumesQuery.data?.data ?? [];
  const isPolling = version && POLL_STATUSES.includes(version.status);

  return (
    <div>
      <PageHeader
        eyebrow="Resumes"
        title="ATS resume generator"
        description="Optimize a parsed resume against a specific job description, with a live ATS score."
        actions={
          <Button onClick={() => setCreateOpen(true)}>
            <Plus className="size-4" aria-hidden />
            Optimize for a job
          </Button>
        }
      />

      <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
        <aside>
          {atsResumesQuery.isLoading ? (
            <SkeletonText lines={4} />
          ) : atsResumes.length === 0 ? (
            <EmptyState icon={FileStack} title="No ATS resumes yet" description="Create one to get a tailored, scored version." />
          ) : (
            <motion.div variants={staggerChildren(0.05)} initial="hidden" animate="visible" className="space-y-2">
              {atsResumes.map((resume) => (
                <motion.button
                  key={resume.id}
                  variants={fadeInUp}
                  onClick={() => {
                    setSelectedAtsResumeId(resume.id);
                    setSelectedVersionId(resume.current_version?.id ?? null);
                  }}
                  className={cn(
                    "block w-full rounded-[var(--radius-lg)] border p-3 text-left transition-colors",
                    resume.id === selectedAtsResumeId ? "border-brand-border bg-brand-soft" : "border-border bg-surface hover:bg-surface-2",
                  )}
                >
                  <p className="truncate text-sm font-medium text-ink">{resume.title}</p>
                  {resume.job_title && <p className="truncate text-xs text-ink-muted">{resume.job_title}{resume.company_name ? ` · ${resume.company_name}` : ""}</p>}
                  {resume.current_version && (
                    <div className="mt-1.5 flex items-center gap-2">
                      <Badge tone="neutral">{formatEnumLabel(resume.current_version.status)}</Badge>
                      {resume.current_version.ats_score != null && <Badge tone="brand">ATS {resume.current_version.ats_score.toFixed(0)}</Badge>}
                    </div>
                  )}
                </motion.button>
              ))}
            </motion.div>
          )}
        </aside>

        <div className="min-w-0">
          {!selectedAtsResumeId ? (
            <EmptyState icon={Sparkles} title="Select or create an ATS resume" description="Pick one from the list, or optimize a new one for a job description." />
          ) : (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center gap-2">
                <Button variant="secondary" size="sm" onClick={() => createVersion.mutate()} isLoading={createVersion.isPending}>
                  <RefreshCw className="size-3.5" aria-hidden />
                  Regenerate
                </Button>
                <Button variant="ghost" size="sm" className="text-danger hover:bg-danger-soft" onClick={() => deleteResume.mutate()}>
                  <Trash2 className="size-3.5" aria-hidden />
                  Delete
                </Button>
                {version && !version.is_current && (
                  <Button variant="ghost" size="sm" onClick={() => activateVersion.mutate(version.id)} isLoading={activateVersion.isPending}>
                    Set as current
                  </Button>
                )}
                {version && versionsQuery.data && versionsQuery.data.length > 1 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-danger hover:bg-danger-soft"
                    onClick={() => deleteVersion.mutate(version.id)}
                    isLoading={deleteVersion.isPending}
                  >
                    <Trash2 className="size-3.5" aria-hidden />
                    Delete version
                  </Button>
                )}
                {versionsQuery.data && versionsQuery.data.length > 1 && (
                  <Select
                    className="ml-auto w-40"
                    value={selectedVersionId ?? ""}
                    onValueChange={setSelectedVersionId}
                    options={versionsQuery.data.map((v) => ({ value: v.id, label: `Version ${v.version_number}${v.is_current ? " (current)" : ""}` }))}
                  />
                )}
              </div>

              {isPolling && (
                <Card className="border-brand-border bg-brand-soft">
                  <CardContent className="flex items-center gap-3 p-4">
                    <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1.2, ease: "linear" }}>
                      <Sparkles className="size-4 text-brand" aria-hidden />
                    </motion.div>
                    <p className="text-sm text-ink">Claude is optimizing this resume — {formatEnumLabel(version!.status)}...</p>
                  </CardContent>
                </Card>
              )}

              {version && version.status === "completed" && (
                <>
                  <Card>
                    <CardContent className="space-y-1 p-5">
                      <h3 className="text-sm font-semibold text-ink">{version.contact.full_name || "—"}</h3>
                      <p className="text-xs text-ink-muted">
                        {[
                          version.contact.phone,
                          version.contact.email,
                          version.contact.linkedin_url,
                          version.contact.github_url,
                          version.contact.portfolio_url,
                          version.contact.location,
                        ]
                          .filter(Boolean)
                          .join("  ·  ") || "No contact details extracted"}
                      </p>
                    </CardContent>
                  </Card>

                  {version.validation_warnings.length > 0 && (
                    <Card className="border-warning-soft bg-warning-soft/40">
                      <CardContent className="space-y-1.5 p-4">
                        <div className="flex items-center gap-2 text-sm font-semibold text-warning">
                          <AlertTriangle className="size-4" aria-hidden />
                          Review before downloading
                        </div>
                        <ul className="list-inside list-disc space-y-0.5 text-xs text-ink-secondary">
                          {version.validation_warnings.map((warning) => (
                            <li key={warning}>{warning}</li>
                          ))}
                        </ul>
                      </CardContent>
                    </Card>
                  )}

                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                    <Card className="flex flex-col items-center gap-2 p-4">
                      <ScoreRing value={version.ats_score ?? 0} size={56} />
                      <p className="text-xs text-ink-muted">ATS score</p>
                    </Card>
                    <Card className="flex flex-col items-center gap-2 p-4">
                      <ScoreRing value={version.keyword_score ?? 0} size={56} />
                      <p className="text-xs text-ink-muted">Keyword score</p>
                    </Card>
                    <Card className="col-span-2 flex flex-col justify-center gap-2 p-4 sm:col-span-2">
                      <div className="flex items-center gap-2">
                        {(["docx", "pdf"] as ATSFileFormat[]).map((format) => (
                          <Button
                            key={format}
                            size="sm"
                            variant="outline"
                            disabled={!version.available_formats.includes(format)}
                            onClick={() => handleDownload(format)}
                          >
                            <Download className="size-3.5" aria-hidden />
                            {format.toUpperCase()}
                          </Button>
                        ))}
                      </div>
                      <p className="text-xs text-ink-muted">Download the optimized file</p>
                    </Card>
                  </div>

                  <Card>
                    <CardContent className="space-y-4 p-5">
                      <div>
                        <h3 className="mb-1.5 text-sm font-semibold text-ink">Optimized summary</h3>
                        <p className="text-sm text-ink-secondary">{version.optimized_summary || "—"}</p>
                      </div>
                      <div className="space-y-2.5">
                        <h3 className="text-sm font-semibold text-ink">Technical skills</h3>
                        {version.grouped_skills.length === 0 ? (
                          <div className="flex flex-wrap gap-1.5">
                            {version.optimized_skills.map((skill) => (
                              <Badge key={skill} tone="brand">{skill}</Badge>
                            ))}
                          </div>
                        ) : (
                          version.grouped_skills.map((group) => (
                            <div key={group.category}>
                              <p className="mb-1 text-xs font-medium text-ink-secondary">{group.category}</p>
                              <div className="flex flex-wrap gap-1.5">
                                {group.skills.map((skill) => (
                                  <Badge key={skill} tone="brand">{skill}</Badge>
                                ))}
                              </div>
                            </div>
                          ))
                        )}
                      </div>
                      <div className="grid gap-4 sm:grid-cols-2">
                        <div>
                          <h3 className="mb-1.5 text-sm font-semibold text-ink">Matched keywords</h3>
                          <div className="flex flex-wrap gap-1.5">
                            {version.matched_keywords.length === 0 ? (
                              <span className="text-xs text-ink-muted">None</span>
                            ) : (
                              version.matched_keywords.map((kw) => <Badge key={kw} tone="success">{kw}</Badge>)
                            )}
                          </div>
                        </div>
                        <div>
                          <h3 className="mb-1.5 text-sm font-semibold text-ink">Missing keywords</h3>
                          <div className="flex flex-wrap gap-1.5">
                            {version.missing_keywords.length === 0 ? (
                              <span className="text-xs text-ink-muted">None</span>
                            ) : (
                              version.missing_keywords.map((kw) => <Badge key={kw} tone="danger">{kw}</Badge>)
                            )}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardContent className="p-5">
                      <h3 className="mb-3 text-sm font-semibold text-ink">Template</h3>
                      <TemplatePicker
                        templates={templates}
                        value={version.template_id}
                        disabled={switchTemplate.isPending}
                        onChange={(newTemplateId) => {
                          if (newTemplateId !== version.template_id) switchTemplate.mutate(newTemplateId);
                        }}
                      />
                      <p className="mt-2 text-xs text-ink-muted">
                        Switching re-renders your DOCX/PDF with the new look — no re-optimization needed.
                      </p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardContent className="p-5">
                      <div className="mb-3 flex items-center justify-between">
                        <h3 className="text-sm font-semibold text-ink">Compare</h3>
                        <Select
                          className="w-48"
                          value={compareTo}
                          onValueChange={setCompareTo}
                          options={[
                            { value: "source", label: "Original resume" },
                            ...(versionsQuery.data ?? [])
                              .filter((v) => v.id !== selectedVersionId)
                              .map((v) => ({ value: v.id, label: `Version ${v.version_number}` })),
                          ]}
                        />
                      </div>
                      {diffQuery.data && (
                        <div className="space-y-3 text-sm leading-relaxed">
                          <p><DiffText segments={diffQuery.data.summary_diff} /></p>
                          <div className="flex flex-wrap gap-3 text-xs">
                            <span className="text-success">+ {diffQuery.data.skills_diff.added.join(", ") || "none"}</span>
                            <span className="text-danger">- {diffQuery.data.skills_diff.removed.join(", ") || "none"}</span>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </>
              )}

              {version && version.status === "failed" && (
                <EmptyState title="Optimization failed" description={version.error_message ?? "Try regenerating this version."} />
              )}
            </div>
          )}
        </div>
      </div>

      <Dialog
        open={createOpen}
        onOpenChange={setCreateOpen}
        title="Optimize a resume for a job"
        description="Pick a parsed resume and paste the target job description."
        footer={
          <>
            <Button variant="ghost" onClick={() => setCreateOpen(false)}>Cancel</Button>
            <Button
              onClick={() => createResume.mutate()}
              isLoading={createResume.isPending}
              disabled={!sourceResumeVersionId || jobDescription.trim().length < 20}
            >
              Optimize resume
            </Button>
          </>
        }
      >
        <div className="space-y-3">
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-ink-secondary">Source resume</label>
            {parsedResumeOptions.length === 0 ? (
              <p className="text-xs text-ink-muted">Upload and parse a resume first.</p>
            ) : (
              <Select value={sourceResumeVersionId} onValueChange={setSourceResumeVersionId} options={parsedResumeOptions} placeholder="Choose a parsed resume" />
            )}
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Input placeholder="Title (optional)" value={title} onChange={(e) => setTitle(e.target.value)} />
            <Input placeholder="Job title (optional)" value={jobTitle} onChange={(e) => setJobTitle(e.target.value)} />
          </div>
          <Input placeholder="Company (optional)" value={companyName} onChange={(e) => setCompanyName(e.target.value)} />
          <Textarea
            placeholder="Paste the target job description here..."
            rows={6}
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
          />
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-ink-secondary">Template</label>
            <TemplatePicker templates={templates} value={templateId} onChange={setTemplateId} />
          </div>
        </div>
      </Dialog>
    </div>
  );
}
