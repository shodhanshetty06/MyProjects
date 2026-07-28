import { motion } from "framer-motion";
import { Bell } from "lucide-react";
import { Link } from "react-router-dom";

import { staggerChildren } from "@/components/motion/variants";
import { Badge } from "@/components/ui/Badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { SkeletonText } from "@/components/ui/Skeleton";
import type { NotificationRead } from "@/features/notification/types";
import { formatRelativeDate } from "@/lib/format";
import { paths } from "@/routes/paths";

interface NotificationsPanelProps {
  notifications: NotificationRead[];
  unreadCount: number;
  isLoading: boolean;
}

export function NotificationsPanel({ notifications, unreadCount, isLoading }: NotificationsPanelProps) {
  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-3 space-y-0">
        <CardTitle className="flex items-center gap-1.5">
          <Bell className="size-4 text-brand" aria-hidden />
          Notifications
        </CardTitle>
        {unreadCount > 0 && <Badge tone="brand">{unreadCount} unread</Badge>}
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <SkeletonText lines={3} />
        ) : notifications.length === 0 ? (
          <EmptyState icon={Bell} title="You're all caught up" description="New notifications will show up here." />
        ) : (
          <motion.ul variants={staggerChildren(0.05)} initial="hidden" animate="visible" className="space-y-1">
            {notifications.map((notification) => (
              <motion.li key={notification.id} variants={{ hidden: { opacity: 0 }, visible: { opacity: 1 } }}>
                <Link
                  to={paths.notifications}
                  className="flex items-start gap-3 rounded-[var(--radius-md)] px-2 py-2 transition-colors hover:bg-surface-2"
                >
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-ink">{notification.title}</p>
                    <p className="line-clamp-1 text-xs text-ink-secondary">{notification.body}</p>
                  </div>
                  <span className="w-16 shrink-0 text-right text-xs text-ink-muted">
                    {formatRelativeDate(notification.created_at)}
                  </span>
                </Link>
              </motion.li>
            ))}
          </motion.ul>
        )}
      </CardContent>
    </Card>
  );
}
