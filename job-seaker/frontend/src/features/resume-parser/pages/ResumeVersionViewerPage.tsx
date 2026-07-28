import { useQuery } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, Download, GraduationCap, Loader2, Sparkles } from "lucide-react";
import { useParams } from "react-router-dom";

import { PageHeader } from "@/components/layout/PageHeader";
import { fadeInUp, staggerChildren } from "@/components/motion/variants";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent } from "@/components/ui/Card";
import { ErrorState } from "@/components/ui/ErrorState";
import { useAuth } from "@/features/auth";
import { getResumeVersion, getResumeVersionDownloadUrl } from "@/features/resume-parser/api";
import { useParsingStageMessage } from "@/features/resume-parser/hooks";
import type { ResumeVersionStatus } from "@/features/resume-parser/types";
import { VersionStatusBadge } from "@/features/resume-parser/components/VersionStatusBadge";

const POLL_STATUSES: ResumeVersionStatus[] = ["uploaded", "parsing"];

export function ResumeVersionViewerPage() {
  const { resumeId, versionId } = useParams<{ resumeId: string; versionId: string }>();
  const { user } = useAuth();
  const userId = user!.id;

  const versionQuery = useQuery({
    queryKey: ["resume-version", userId, resumeId, versionId],
    queryFn: () => getResumeVersion(userId, resumeId as string, versionId as string),
    enabled: !!resumeId && !!versionId,
    refetchInterval: (query) => (query.state.data && POLL_STATUSES.includes(query.state.data.status) ? 2500 : false),
  });

  const version = versionQuery.data;
  const stageMessage = useParsingStageMessage(version?.status);

  async function handleDownload() {
    if (!resumeId || !versionId) return;
    const url = await getResumeVersionDownloadUrl(userId, resumeId, versionId);
    window.open(url, "_blank", "noopener,noreferrer");
  }

  if (versionQuery.isLoading || !version) {
    return (
      <div className="flex min-h-[40vh] items-center justify-center">
        <Loader2 className="size-6 animate-spin text-brand" aria-hidden />
      </div>
    );
  }

  if (versionQuery.isError) {
    return <ErrorState title="Couldn't load this version" />;
  }

  if (version.status === "uploaded" || version.status === "parsing") {
    return (
      <div className="mx-auto max-w-md py-16 text-center">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ repeat: Infinity, duration: 1.6, ease: "linear" }}
          className="mx-auto mb-4 flex size-14 items-center justify-center rounded-full bg-brand-soft text-brand"
        >
          <Sparkles className="size-6" aria-hidden />
        </motion.div>
        <h1 className="text-lg font-semibold text-ink">Parsing your resume...</h1>
        <p className="mt-1 text-sm text-ink-secondary">
          {version.original_filename} is being read into structured skills, experience, and education.
        </p>
        <div className="mt-2 h-5 overflow-hidden">
          <AnimatePresence mode="wait">
            <motion.p
              key={stageMessage}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.25 }}
              className="text-sm font-medium text-brand"
            >
              {stageMessage}
            </motion.p>
          </AnimatePresence>
        </div>
      </div>
    );
  }

  if (version.status === "failed") {
    return (
      <ErrorState
        title="Parsing failed"
        description={version.error_message ?? "We couldn't parse this file. Try uploading a different PDF or DOCX."}
      />
    );
  }

  const data = version.parsed_data;

  return (
    <div className="mx-auto max-w-3xl">
      <PageHeader
        eyebrow={`Version ${version.version_number}`}
        title={version.original_filename}
        actions={
          <div className="flex items-center gap-2">
            <VersionStatusBadge status={version.status} />
            <Button variant="secondary" onClick={handleDownload}>
              <Download className="size-4" aria-hidden />
              Download
            </Button>
          </div>
        }
      />

      {!data ? (
        <ErrorState title="No parsed data available" />
      ) : (
        <motion.div variants={staggerChildren(0.06)} initial="hidden" animate="visible" className="space-y-4">
          {data.summary && (
            <motion.div variants={fadeInUp}>
              <Card>
                <CardContent className="p-5">
                  <h2 className="mb-2 text-sm font-semibold text-ink">Summary</h2>
                  <p className="text-sm leading-relaxed text-ink-secondary">{data.summary}</p>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {data.skills.length > 0 && (
            <motion.div variants={fadeInUp}>
              <Card>
                <CardContent className="p-5">
                  <h2 className="mb-3 text-sm font-semibold text-ink">Skills</h2>
                  <div className="flex flex-wrap gap-1.5">
                    {data.skills.map((skill) => (
                      <Badge key={skill} tone="brand">
                        {skill}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {data.experience.length > 0 && (
            <motion.div variants={fadeInUp}>
              <Card>
                <CardContent className="p-5">
                  <h2 className="mb-3 text-sm font-semibold text-ink">Experience</h2>
                  <div className="space-y-4">
                    {data.experience.map((exp, i) => (
                      <div key={i} className="relative border-l-2 border-border pl-4">
                        <div className="absolute -left-[5px] top-1 size-2 rounded-full bg-brand" />
                        <p className="text-sm font-medium text-ink">
                          {exp.title} · {exp.company}
                        </p>
                        <p className="text-xs text-ink-muted">
                          {exp.start_date ?? "?"} - {exp.is_current ? "Present" : (exp.end_date ?? "?")}
                          {exp.location ? ` · ${exp.location}` : ""}
                        </p>
                        {exp.description && <p className="mt-1 text-xs text-ink-secondary">{exp.description}</p>}
                        {exp.highlights.length > 0 && (
                          <ul className="mt-1 list-disc space-y-0.5 pl-4 text-xs text-ink-secondary">
                            {exp.highlights.map((h, hi) => (
                              <li key={hi}>{h}</li>
                            ))}
                          </ul>
                        )}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {data.education.length > 0 && (
            <motion.div variants={fadeInUp}>
              <Card>
                <CardContent className="p-5">
                  <h2 className="mb-3 flex items-center gap-1.5 text-sm font-semibold text-ink">
                    <GraduationCap className="size-4 text-brand" aria-hidden />
                    Education
                  </h2>
                  <div className="space-y-2">
                    {data.education.map((edu, i) => (
                      <div key={i}>
                        <p className="text-sm font-medium text-ink">{edu.institution}</p>
                        <p className="text-xs text-ink-muted">
                          {[edu.degree, edu.field_of_study].filter(Boolean).join(", ")}
                          {edu.start_date && ` · ${edu.start_date} - ${edu.end_date ?? "Present"}`}
                        </p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {data.projects.length > 0 && (
            <motion.div variants={fadeInUp}>
              <Card>
                <CardContent className="p-5">
                  <h2 className="mb-3 text-sm font-semibold text-ink">Projects</h2>
                  <div className="space-y-3">
                    {data.projects.map((project, i) => (
                      <div key={i}>
                        <p className="text-sm font-medium text-ink">{project.name}</p>
                        {project.description && <p className="text-xs text-ink-secondary">{project.description}</p>}
                        {project.technologies.length > 0 && (
                          <div className="mt-1 flex flex-wrap gap-1">
                            {project.technologies.map((tech) => (
                              <Badge key={tech} tone="neutral" className="text-[10px]">
                                {tech}
                              </Badge>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {(data.certifications.length > 0 || data.languages.length > 0 || data.achievements.length > 0) && (
            <motion.div variants={fadeInUp} className="grid gap-4 sm:grid-cols-2">
              {data.certifications.length > 0 && (
                <Card>
                  <CardContent className="p-5">
                    <h2 className="mb-2 text-sm font-semibold text-ink">Certifications</h2>
                    <ul className="space-y-1 text-xs text-ink-secondary">
                      {data.certifications.map((cert, i) => (
                        <li key={i}>
                          {cert.name}
                          {cert.issuer ? ` · ${cert.issuer}` : ""}
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}
              {data.languages.length > 0 && (
                <Card>
                  <CardContent className="p-5">
                    <h2 className="mb-2 text-sm font-semibold text-ink">Languages</h2>
                    <ul className="space-y-1 text-xs text-ink-secondary">
                      {data.languages.map((lang, i) => (
                        <li key={i}>
                          {lang.language}
                          {lang.proficiency ? ` · ${lang.proficiency}` : ""}
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}
            </motion.div>
          )}

          {data.achievements.length === 0 &&
            data.skills.length === 0 &&
            data.experience.length === 0 &&
            data.education.length === 0 && (
              <ErrorState
                title="Parsed, but nothing extracted"
                description="We couldn't confidently extract structured content from this file."
                action={<AlertTriangle className="size-4 text-warning" aria-hidden />}
              />
            )}
        </motion.div>
      )}
    </div>
  );
}
