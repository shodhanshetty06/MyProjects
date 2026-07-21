import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Mail, MailX, Plug, Unplug } from "lucide-react";

import { PageHeader } from "@/components/layout/PageHeader";
import { staggerChildren } from "@/components/motion/variants";
import { Badge, type BadgeTone } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { SkeletonText } from "@/components/ui/Skeleton";
import {
  disconnectGmailAccount,
  getGmailConnectUrl,
  listGmailAccounts,
  listTrackedEmails,
  resolveTrackedEmail,
} from "@/features/email-tracking/api";
import type { EmailClassification } from "@/features/email-tracking/types";
import { useAuth } from "@/features/auth";
import { formatEnumLabel, formatRelativeDate } from "@/lib/format";
import { toast } from "@/stores/toastStore";

const CLASSIFICATION_TONE: Record<EmailClassification, BadgeTone> = {
  interview: "brand",
  assessment: "warning",
  offer: "success",
  rejection: "danger",
  not_relevant: "neutral",
};

export function TrackerEmailsPage() {
  const { user } = useAuth();
  const userId = user!.id;
  const queryClient = useQueryClient();

  const accountsQuery = useQuery({
    queryKey: ["gmail-accounts", userId],
    queryFn: () => listGmailAccounts(userId),
  });

  const emailsQuery = useQuery({
    queryKey: ["tracked-emails", userId],
    queryFn: () => listTrackedEmails(userId, { page_size: 30 }),
  });

  const connectMutation = useMutation({
    mutationFn: () => getGmailConnectUrl(userId),
    onSuccess: (url) => {
      window.location.href = url;
    },
    onError: () => toast.danger("Couldn't start Gmail connection"),
  });

  const disconnectMutation = useMutation({
    mutationFn: (accountId: string) => disconnectGmailAccount(userId, accountId),
    onSuccess: () => {
      toast.success("Gmail account disconnected");
      void queryClient.invalidateQueries({ queryKey: ["gmail-accounts", userId] });
    },
  });

  const dismissMutation = useMutation({
    mutationFn: (emailId: string) => resolveTrackedEmail(userId, emailId, { dismiss: true }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["tracked-emails", userId] });
    },
  });

  const accounts = accountsQuery.data ?? [];
  const emails = emailsQuery.data?.data ?? [];

  return (
    <div>
      <PageHeader
        eyebrow="Applications"
        title="Tracked emails"
        description="Connect Gmail to automatically match interview, assessment, and offer emails to your applications."
      />

      <Card className="mb-6">
        <CardContent className="p-5">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-ink">Connected accounts</h2>
            <Button size="sm" onClick={() => connectMutation.mutate()} isLoading={connectMutation.isPending}>
              <Plug className="size-3.5" aria-hidden />
              Connect Gmail
            </Button>
          </div>
          {accountsQuery.isLoading ? (
            <SkeletonText lines={2} />
          ) : accounts.length === 0 ? (
            <p className="text-xs text-ink-muted">No Gmail account connected yet.</p>
          ) : (
            <ul className="space-y-2">
              {accounts.map((account) => (
                <li key={account.id} className="flex items-center gap-3 rounded-[var(--radius-md)] border border-border p-2.5">
                  <Mail className="size-4 text-ink-muted" aria-hidden />
                  <span className="flex-1 truncate text-sm text-ink">{account.email_address}</span>
                  <Badge tone={account.status === "connected" ? "success" : "warning"}>{formatEnumLabel(account.status)}</Badge>
                  <Button variant="ghost" size="sm" onClick={() => disconnectMutation.mutate(account.id)}>
                    <Unplug className="size-3.5" aria-hidden />
                  </Button>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      <h2 className="mb-3 text-sm font-semibold text-ink">Recent emails</h2>
      {emailsQuery.isLoading ? (
        <SkeletonText lines={5} />
      ) : emails.length === 0 ? (
        <EmptyState icon={Mail} title="No tracked emails yet" description="Connect a Gmail account to start matching inbound emails." />
      ) : (
        <motion.div variants={staggerChildren(0.04)} initial="hidden" animate="visible" className="space-y-2">
          {emails.map((email) => (
            <motion.div key={email.id} variants={{ hidden: { opacity: 0, y: 6 }, visible: { opacity: 1, y: 0 } }}>
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium text-ink">{email.subject}</p>
                      <p className="truncate text-xs text-ink-muted">{email.sender}</p>
                    </div>
                    <Badge tone={CLASSIFICATION_TONE[email.classification]}>{formatEnumLabel(email.classification)}</Badge>
                  </div>
                  <p className="mt-1.5 line-clamp-2 text-xs text-ink-secondary">{email.snippet}</p>
                  <div className="mt-2 flex items-center justify-between">
                    <span className="text-[11px] text-ink-muted">{formatRelativeDate(email.received_at)}</span>
                    {email.match_status === "needs_review" && (
                      <Button variant="ghost" size="sm" onClick={() => dismissMutation.mutate(email.id)}>
                        <MailX className="size-3.5" aria-hidden />
                        Dismiss
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      )}
    </div>
  );
}
