import { useEffect, useRef, useState } from "react";

import { buildWorkspaceSocketUrl } from "@/features/workspace/api";
import type { WorkspaceRealtimeEvent } from "@/features/workspace/types";

export type SocketStatus = "idle" | "connecting" | "open" | "closed";

// Capped exponential backoff so a server restart (all clients dropped at
// once) doesn't cause every open tab to reconnect in lockstep — each retry
// waits longer, up to 30s, before trying again.
const RECONNECT_BASE_DELAY_MS = 1000;
const RECONNECT_MAX_DELAY_MS = 30_000;

/**
 * Subscribes to `workspace.router`'s `/workspace/ws` endpoint for one user
 * and reports each parsed event via `onEvent` (typically used to invalidate
 * the `["workspace"]` React Query key) plus a connection-status flag for the
 * UI. One socket covers pipeline progress, match/recommendation updates,
 * ATS-resume-ready, and application-status-changed events — see
 * `app/modules/workspace/router.py`'s docstring for why it's just one.
 *
 * Reconnects automatically with capped exponential backoff if the connection
 * drops (server restart, network blip) so the page doesn't silently go stale
 * until the user manually navigates away and back.
 */
export function useWorkspaceSocket(
  userId: string | null,
  onEvent: (event: WorkspaceRealtimeEvent) => void,
): SocketStatus {
  const [status, setStatus] = useState<SocketStatus>("idle");
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  useEffect(() => {
    if (!userId) {
      setStatus("idle");
      return;
    }

    let cancelled = false;
    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
    let attempt = 0;

    function connect() {
      if (cancelled) return;
      setStatus("connecting");
      socket = new WebSocket(buildWorkspaceSocketUrl(userId as string));

      socket.onopen = () => {
        attempt = 0;
        setStatus("open");
      };
      socket.onclose = () => {
        if (cancelled) return;
        setStatus("closed");
        const delay = Math.min(RECONNECT_BASE_DELAY_MS * 2 ** attempt, RECONNECT_MAX_DELAY_MS);
        attempt += 1;
        reconnectTimer = setTimeout(connect, delay);
      };
      socket.onerror = () => socket?.close();
      socket.onmessage = (message) => {
        try {
          onEventRef.current(JSON.parse(message.data) as WorkspaceRealtimeEvent);
        } catch {
          // Ignore malformed frames (e.g. keepalive pings that aren't JSON).
        }
      };
    }

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [userId]);

  return status;
}
