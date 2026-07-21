import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { FileText } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";

import { PageHeader } from "@/components/layout/PageHeader";
import { staggerChildren } from "@/components/motion/variants";
import { Card } from "@/components/ui/Card";
import { Dropzone } from "@/components/ui/Dropzone";
import { EmptyState } from "@/components/ui/EmptyState";
import { ErrorState } from "@/components/ui/ErrorState";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { SkeletonCard } from "@/components/ui/Skeleton";
import { useAuth } from "@/features/auth";
import { listResumes, uploadResume } from "@/features/resume-parser/api";
import { VersionStatusBadge } from "@/features/resume-parser/components/VersionStatusBadge";
import { formatRelativeDate } from "@/lib/format";
import { paths } from "@/routes/paths";
import { toast } from "@/stores/toastStore";

export function ResumeManagerPage() {
  const { user } = useAuth();
  const userId = user!.id;
  const queryClient = useQueryClient();
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);

  const resumesQuery = useQuery({
    queryKey: ["resumes", userId],
    queryFn: () => listResumes(userId, 1, 50),
  });

  const uploadMutation = useMutation({
    mutationFn: (file: File) =>
      uploadResume(userId, file, file.name.replace(/\.(pdf|docx)$/i, ""), (event) => {
        if (event.total) setUploadProgress(Math.round((event.loaded / event.total) * 100));
      }),
    onSuccess: () => {
      toast.success("Resume uploaded", "Parsing has started.");
      setUploadProgress(null);
      void queryClient.invalidateQueries({ queryKey: ["resumes", userId] });
    },
    onError: () => {
      setUploadProgress(null);
      toast.danger("Upload failed", "Try a PDF or DOCX under 10MB.");
    },
  });

  const resumes = resumesQuery.data?.data ?? [];

  return (
    <div>
      <PageHeader
        eyebrow="Resumes"
        title="Resume manager"
        description="Upload, parse, and version every resume you use in your search."
      />

      <Card className="mb-6 p-5">
        <Dropzone
          onFile={(file) => uploadMutation.mutate(file)}
          disabled={uploadMutation.isPending}
          label={uploadMutation.isPending ? "Uploading..." : "Drop a resume here, or click to browse"}
        />
        {uploadProgress !== null && (
          <div className="mt-3">
            <ProgressBar value={uploadProgress} />
          </div>
        )}
      </Card>

      {resumesQuery.isLoading ? (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      ) : resumesQuery.isError ? (
        <ErrorState description="Couldn't load your resumes." />
      ) : resumes.length === 0 ? (
        <EmptyState icon={FileText} title="No resumes yet" description="Upload one above to get started - we'll parse it automatically." />
      ) : (
        <motion.div variants={staggerChildren(0.05)} initial="hidden" animate="visible" className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {resumes.map((resume) => (
            <motion.div key={resume.id} variants={{ hidden: { opacity: 0, y: 8 }, visible: { opacity: 1, y: 0 } }}>
              <Link to={paths.resumeDetail(resume.id)}>
                <Card className="flex flex-col gap-2 p-4 transition-all hover:-translate-y-0.5 hover:border-brand-border hover:shadow-sm">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex size-9 items-center justify-center rounded-[var(--radius-md)] bg-brand-soft text-brand">
                      <FileText className="size-4" aria-hidden />
                    </div>
                    {resume.current_version && <VersionStatusBadge status={resume.current_version.status} />}
                  </div>
                  <p className="truncate text-sm font-semibold text-ink">{resume.title}</p>
                  <p className="text-xs text-ink-muted">
                    {resume.current_version ? `Version ${resume.current_version.version_number}` : "No versions"} · Updated{" "}
                    {formatRelativeDate(resume.updated_at)}
                  </p>
                </Card>
              </Link>
            </motion.div>
          ))}
        </motion.div>
      )}
    </div>
  );
}
