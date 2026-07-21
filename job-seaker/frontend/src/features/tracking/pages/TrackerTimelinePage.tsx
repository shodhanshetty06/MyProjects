import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { ArrowRight, ExternalLink, MapPin, MessageSquarePlus, Send, Trash2 } from "lucide-react";
import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { PageHeader } from "@/components/layout/PageHeader";
import { staggerChildren } from "@/components/motion/variants";
import { Button } from "@/components/ui/Button";
import { Card, CardContent } from "@/components/ui/Card";
import { ErrorState } from "@/components/ui/ErrorState";
import { Select } from "@/components/ui/Select";
import { SkeletonText } from "@/components/ui/Skeleton";
import { Textarea } from "@/components/ui/Textarea";
import { useAuth } from "@/features/auth";
import {
  changeApplicationStatus,
  createApplicationNote,
  deleteApplication,
  getApplication,
  getApplicationTimeline,
} from "@/features/tracking/api";
import { ApplicationStatusBadge } from "@/features/tracking/components/ApplicationStatusBadge";
import {
  APPLICATION_STATUSES,
  type ApplicationNoteRead,
  type ApplicationStatus,
  type ApplicationStatusHistoryRead,
} from "@/features/tracking/types";
import { formatEnumLabel, formatRelativeDate } from "@/lib/format";
import { paths } from "@/routes/paths";
import { toast } from "@/stores/toastStore";

export function TrackerTimelinePage() {
  const { applicationId } = useParams<{ applicationId: string }>();
  const { user } = useAuth();
  const userId = user!.id;
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [noteText, setNoteText] = useState("");

  const applicationQuery = useQuery({
    queryKey: ["application", userId, applicationId],
    queryFn: () => getApplication(userId, applicationId as string),
    enabled: !!applicationId,
  });

  const timelineQuery = useQuery({
    queryKey: ["application-timeline", userId, applicationId],
    queryFn: () => getApplicationTimeline(userId, applicationId as string),
    enabled: !!applicationId,
  });

  const invalidate = () => {
    void queryClient.invalidateQueries({ queryKey: ["application", userId, applicationId] });
    void queryClient.invalidateQueries({ queryKey: ["application-timeline", userId, applicationId] });
    void queryClient.invalidateQueries({ queryKey: ["applications", userId] });
    void queryClient.invalidateQueries({ queryKey: ["application-statistics", userId] });
  };

  const statusMutation = useMutation({
    mutationFn: (status: ApplicationStatus) => changeApplicationStatus(userId, applicationId as string, { status }),
    onSuccess: () => {
      toast.success("Status updated");
      invalidate();
    },
  });

  const noteMutation = useMutation({
    mutationFn: (content: string) => createApplicationNote(userId, applicationId as string, content),
    onSuccess: () => {
      setNoteText("");
      invalidate();
    },
  });

  const deleteMutation = useMutation({
    mutationFn: () => deleteApplication(userId, applicationId as string),
    onSuccess: () => {
      toast.success("Application removed");
      navigate(paths.tracker);
    },
  });

  if (applicationQuery.isLoading) return <SkeletonText lines={6} />;
  if (applicationQuery.isError || !applicationQuery.data) return <ErrorState title="Application not found" />;

  const app = applicationQuery.data;
  const timeline = timelineQuery.data ?? [];

  return (
    <div className="mx-auto max-w-2xl">
      <PageHeader
        eyebrow="Application"
        title={app.job_title}
        description={app.company_name}
        actions={
          <Button variant="ghost" className="text-danger hover:bg-danger-soft" onClick={() => deleteMutation.mutate()}>
            <Trash2 className="size-4" aria-hidden />
            Remove
          </Button>
        }
      />

      <Card className="mb-6">
        <CardContent className="flex flex-wrap items-center gap-4 p-5">
          <ApplicationStatusBadge status={app.status} />
          <Select
            className="w-40"
            value={app.status}
            onValueChange={(v) => statusMutation.mutate(v as ApplicationStatus)}
            options={APPLICATION_STATUSES.map((s) => ({ value: s, label: formatEnumLabel(s) }))}
          />
          {app.location && (
            <span className="flex items-center gap-1 text-xs text-ink-muted">
              <MapPin className="size-3" aria-hidden />
              {app.location}
            </span>
          )}
          {app.job_url && (
            <a href={app.job_url} target="_blank" rel="noreferrer" className="ml-auto flex items-center gap-1 text-xs font-medium text-brand hover:underline">
              View posting
              <ExternalLink className="size-3" aria-hidden />
            </a>
          )}
        </CardContent>
      </Card>

      <Card className="mb-6">
        <CardContent className="p-5">
          <h2 className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-ink">
            <MessageSquarePlus className="size-4 text-brand" aria-hidden />
            Add a note
          </h2>
          <Textarea rows={3} value={noteText} onChange={(e) => setNoteText(e.target.value)} placeholder="Interview notes, follow-up reminders, contacts..." />
          <div className="mt-2 flex justify-end">
            <Button
              size="sm"
              disabled={!noteText.trim()}
              isLoading={noteMutation.isPending}
              onClick={() => noteMutation.mutate(noteText)}
            >
              <Send className="size-3.5" aria-hidden />
              Add note
            </Button>
          </div>
        </CardContent>
      </Card>

      <h2 className="mb-3 text-sm font-semibold text-ink">Timeline</h2>
      {timelineQuery.isLoading ? (
        <SkeletonText lines={4} />
      ) : timeline.length === 0 ? (
        <p className="text-sm text-ink-muted">No activity yet.</p>
      ) : (
        <motion.ol variants={staggerChildren(0.05)} initial="hidden" animate="visible" className="space-y-4">
          {timeline.map((entry, i) => (
            <motion.li key={i} variants={{ hidden: { opacity: 0, x: -8 }, visible: { opacity: 1, x: 0 } }} className="relative border-l-2 border-border pl-4">
              <div className="absolute -left-[5px] top-1 size-2 rounded-full bg-brand" />
              {entry.type === "status_change" ? (
                (() => {
                  const data = entry.data as ApplicationStatusHistoryRead;
                  return (
                    <p className="text-sm text-ink">
                      {data.from_status ? (
                        <>
                          <ApplicationStatusBadge status={data.from_status} /> <ArrowRight className="inline size-3" aria-hidden />{" "}
                        </>
                      ) : null}
                      <ApplicationStatusBadge status={data.to_status} />
                      {data.note && <span className="ml-2 text-ink-secondary">{data.note}</span>}
                    </p>
                  );
                })()
              ) : (
                <p className="text-sm text-ink-secondary">{(entry.data as ApplicationNoteRead).content}</p>
              )}
              <p className="mt-0.5 text-xs text-ink-muted">{formatRelativeDate(entry.occurred_at)}</p>
            </motion.li>
          ))}
        </motion.ol>
      )}
    </div>
  );
}
