import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Bell, BellOff, Check, CheckCheck, Settings } from "lucide-react";
import { useState } from "react";

import { PageHeader } from "@/components/layout/PageHeader";
import { staggerChildren } from "@/components/motion/variants";
import { Button } from "@/components/ui/Button";
import { Card, CardContent } from "@/components/ui/Card";
import { Dialog } from "@/components/ui/Dialog";
import { EmptyState } from "@/components/ui/EmptyState";
import { SkeletonText } from "@/components/ui/Skeleton";
import { Switch } from "@/components/ui/Switch";
import { useAuth } from "@/features/auth";
import {
  getNotificationPreferences,
  listNotifications,
  markAllNotificationsRead,
  markNotificationRead,
  updateNotificationPreferences,
} from "@/features/notification/api";
import {
  NOTIFICATION_CHANNELS,
  NOTIFICATION_EVENT_TYPES,
  type NotificationPreferenceUpdate,
} from "@/features/notification/types";
import { cn } from "@/lib/cn";
import { formatEnumLabel, formatRelativeDate } from "@/lib/format";

export function NotificationPage() {
  const { user } = useAuth();
  const userId = user!.id;
  const queryClient = useQueryClient();
  const [prefsOpen, setPrefsOpen] = useState(false);

  const notificationsQuery = useQuery({
    queryKey: ["notifications", userId],
    queryFn: () => listNotifications(userId, false, 1, 40),
    refetchInterval: 45_000,
  });

  const preferencesQuery = useQuery({
    queryKey: ["notification-preferences", userId],
    queryFn: () => getNotificationPreferences(userId),
    enabled: prefsOpen,
  });

  const readMutation = useMutation({
    mutationFn: (id: string) => markNotificationRead(userId, id),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ["notifications", userId] }),
  });

  const readAllMutation = useMutation({
    mutationFn: () => markAllNotificationsRead(userId),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ["notifications", userId] }),
  });

  const preferencesMutation = useMutation({
    mutationFn: (preferences: NotificationPreferenceUpdate[]) => updateNotificationPreferences(userId, preferences),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ["notification-preferences", userId] }),
  });

  function togglePreference(eventType: (typeof NOTIFICATION_EVENT_TYPES)[number], channel: (typeof NOTIFICATION_CHANNELS)[number], current: boolean) {
    const existing = preferencesQuery.data ?? [];
    const next: NotificationPreferenceUpdate[] = NOTIFICATION_EVENT_TYPES.flatMap((et) =>
      NOTIFICATION_CHANNELS.map((ch) => {
        const match = existing.find((p) => p.event_type === et && p.channel === ch);
        const enabled = et === eventType && ch === channel ? !current : (match?.enabled ?? true);
        return { event_type: et, channel: ch, enabled };
      }),
    );
    preferencesMutation.mutate(next);
  }

  const notifications = notificationsQuery.data?.data ?? [];
  const unreadCount = notifications.filter((n) => !n.read_at).length;

  return (
    <div className="mx-auto max-w-2xl">
      <PageHeader
        eyebrow="Notifications"
        title="Notifications"
        description={unreadCount > 0 ? `${unreadCount} unread` : "You're all caught up"}
        actions={
          <div className="flex items-center gap-2">
            <Button variant="secondary" onClick={() => setPrefsOpen(true)}>
              <Settings className="size-4" aria-hidden />
              Preferences
            </Button>
            {unreadCount > 0 && (
              <Button variant="ghost" onClick={() => readAllMutation.mutate()} isLoading={readAllMutation.isPending}>
                <CheckCheck className="size-4" aria-hidden />
                Mark all read
              </Button>
            )}
          </div>
        }
      />

      {notificationsQuery.isLoading ? (
        <SkeletonText lines={6} />
      ) : notifications.length === 0 ? (
        <EmptyState icon={BellOff} title="No notifications" description="You'll see status changes and account alerts here." />
      ) : (
        <motion.div variants={staggerChildren(0.04)} initial="hidden" animate="visible" className="space-y-2">
          {notifications.map((n) => (
            <motion.div key={n.id} variants={{ hidden: { opacity: 0, y: 6 }, visible: { opacity: 1, y: 0 } }}>
              <Card className={cn(!n.read_at && "border-brand-border bg-brand-soft")}>
                <CardContent className="flex items-start gap-3 p-4">
                  <div
                    className={cn(
                      "mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-full",
                      n.read_at ? "bg-surface-2 text-ink-muted" : "bg-brand text-brand-ink",
                    )}
                  >
                    <Bell className="size-3.5" aria-hidden />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-ink">{n.title}</p>
                    <p className="mt-0.5 text-sm text-ink-secondary">{n.body}</p>
                    <p className="mt-1 text-[11px] text-ink-muted">{formatRelativeDate(n.created_at)}</p>
                  </div>
                  {!n.read_at && (
                    <button
                      type="button"
                      onClick={() => readMutation.mutate(n.id)}
                      className="shrink-0 rounded-[var(--radius-sm)] p-1.5 text-ink-muted transition-colors hover:bg-surface-2 hover:text-ink"
                      aria-label="Mark as read"
                    >
                      <Check className="size-4" aria-hidden />
                    </button>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      )}

      <Dialog
        open={prefsOpen}
        onOpenChange={setPrefsOpen}
        title="Notification preferences"
        description="Choose which channels each type of update goes to."
      >
        {preferencesQuery.isLoading ? (
          <SkeletonText lines={4} />
        ) : (
          <div className="space-y-4">
            {NOTIFICATION_EVENT_TYPES.map((eventType) => (
              <div key={eventType}>
                <p className="mb-2 text-sm font-medium text-ink">{formatEnumLabel(eventType)}</p>
                <div className="flex items-center gap-4">
                  {NOTIFICATION_CHANNELS.map((channel) => {
                    const pref = preferencesQuery.data?.find((p) => p.event_type === eventType && p.channel === channel);
                    const enabled = pref?.enabled ?? true;
                    return (
                      <label key={channel} className="flex items-center gap-1.5 text-xs text-ink-secondary">
                        <Switch checked={enabled} onCheckedChange={() => togglePreference(eventType, channel, enabled)} />
                        {formatEnumLabel(channel)}
                      </label>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}
      </Dialog>
    </div>
  );
}
