import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { CheckCircle2, FileText, Star, Trash2, UploadCloud } from "lucide-react";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";

import { PageHeader } from "@/components/layout/PageHeader";
import { staggerChildren } from "@/components/motion/variants";
import { Button } from "@/components/ui/Button";
import { Card, CardContent } from "@/components/ui/Card";
import { Dropzone } from "@/components/ui/Dropzone";
import { ErrorState } from "@/components/ui/ErrorState";
import { SkeletonText } from "@/components/ui/Skeleton";
import { useAuth } from "@/features/auth";
import {
  activateResumeVersion,
  deleteResumeVersion,
  getResume,
  listResumeVersions,
  uploadResumeVersion,
} from "@/features/resume-parser/api";
import { VersionStatusBadge } from "@/features/resume-parser/components/VersionStatusBadge";
import { formatRelativeDate } from "@/lib/format";
import { paths } from "@/routes/paths";
import { toast } from "@/stores/toastStore";

export function ResumeDetailPage() {
  const { resumeId } = useParams<{ resumeId: string }>();
  const { user } = useAuth();
  const userId = user!.id;
  const queryClient = useQueryClient();
  const [showUpload, setShowUpload] = useState(false);

  const resumeQuery = useQuery({
    queryKey: ["resume", userId, resumeId],
    queryFn: () => getResume(userId, resumeId as string),
    enabled: !!resumeId,
  });

  const versionsQuery = useQuery({
    queryKey: ["resume-versions", userId, resumeId],
    queryFn: () => listResumeVersions(userId, resumeId as string),
    enabled: !!resumeId,
  });

  const invalidate = () => {
    void queryClient.invalidateQueries({ queryKey: ["resume-versions", userId, resumeId] });
    void queryClient.invalidateQueries({ queryKey: ["resume", userId, resumeId] });
    void queryClient.invalidateQueries({ queryKey: ["resumes", userId] });
  };

  const uploadVersionMutation = useMutation({
    mutationFn: (file: File) => uploadResumeVersion(userId, resumeId as string, file),
    onSuccess: () => {
      toast.success("New version uploaded", "Parsing has started.");
      setShowUpload(false);
      invalidate();
    },
    onError: () => toast.danger("Upload failed", "Try a PDF or DOCX under 10MB."),
  });

  const activateMutation = useMutation({
    mutationFn: (versionId: string) => activateResumeVersion(userId, resumeId as string, versionId),
    onSuccess: () => {
      toast.success("Version activated");
      invalidate();
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (versionId: string) => deleteResumeVersion(userId, resumeId as string, versionId),
    onSuccess: () => {
      toast.success("Version deleted");
      invalidate();
    },
  });

  if (resumeQuery.isLoading) {
    return <SkeletonText lines={6} />;
  }

  if (resumeQuery.isError || !resumeQuery.data) {
    return <ErrorState title="Resume not found" />;
  }

  const resume = resumeQuery.data;
  const versions = versionsQuery.data ?? [];
  const rid = resumeId as string;

  return (
    <div>
      <PageHeader
        eyebrow="Resumes"
        title={resume.title}
        description="Manage versions of this resume - upload a refresh any time and activate whichever fits best."
        actions={
          <Button variant="secondary" onClick={() => setShowUpload((v) => !v)}>
            <UploadCloud className="size-4" aria-hidden />
            Upload new version
          </Button>
        }
      />

      {showUpload && (
        <Card className="mb-6 p-5">
          <Dropzone onFile={(file) => uploadVersionMutation.mutate(file)} disabled={uploadVersionMutation.isPending} />
        </Card>
      )}

      {versionsQuery.isLoading ? (
        <SkeletonText lines={4} />
      ) : (
        <motion.div variants={staggerChildren(0.05)} initial="hidden" animate="visible" className="space-y-2">
          {versions.map((version) => (
            <motion.div key={version.id} variants={{ hidden: { opacity: 0, x: -8 }, visible: { opacity: 1, x: 0 } }}>
              <Card>
                <CardContent className="flex items-center gap-3 p-4">
                  <div className="flex size-9 shrink-0 items-center justify-center rounded-[var(--radius-md)] bg-surface-2 text-ink-muted">
                    <FileText className="size-4" aria-hidden />
                  </div>
                  <div className="min-w-0 flex-1">
                    <Link
                      to={paths.resumeVersion(rid, version.id)}
                      className="flex items-center gap-2 text-sm font-medium text-ink hover:text-brand"
                    >
                      Version {version.version_number}
                      {version.is_current && <Star className="size-3.5 fill-warning text-warning" aria-hidden />}
                    </Link>
                    <p className="truncate text-xs text-ink-muted">
                      {version.original_filename} · {formatRelativeDate(version.created_at)}
                    </p>
                  </div>
                  <VersionStatusBadge status={version.status} />
                  {!version.is_current && (
                    <Button variant="ghost" size="sm" onClick={() => activateMutation.mutate(version.id)}>
                      <CheckCircle2 className="size-3.5" aria-hidden />
                      Activate
                    </Button>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => deleteMutation.mutate(version.id)}
                    className="text-danger hover:bg-danger-soft"
                  >
                    <Trash2 className="size-3.5" aria-hidden />
                  </Button>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      )}
    </div>
  );
}
