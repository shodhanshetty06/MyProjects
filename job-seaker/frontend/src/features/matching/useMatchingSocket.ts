import { useEffect, useRef, useState } from "react";

import { buildMatchingSocketUrl } from "@/features/matching/api";
import type { MatchingRealtimeEvent } from "@/features/matching/types";

export type SocketStatus = "idle" | "connecting" | "open" | "closed";

/**
 * Subscribes to `matching.router`'s `/matching/ws` endpoint for one user and
 * reports each parsed event via `onEvent` (typically used to invalidate the
 * relevant React Query keys) plus a connection-status flag for the UI.
 */
export function useMatchingSocket(
  userId: string | null,
  onEvent: (event: MatchingRealtimeEvent) => void,
): SocketStatus {
  const [status, setStatus] = useState<SocketStatus>("idle");
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  useEffect(() => {
    if (!userId) {
      setStatus("idle");
      return;
    }

    setStatus("connecting");
    const socket = new WebSocket(buildMatchingSocketUrl(userId));

    socket.onopen = () => setStatus("open");
    socket.onclose = () => setStatus("closed");
    socket.onerror = () => setStatus("closed");
    socket.onmessage = (message) => {
      try {
        onEventRef.current(JSON.parse(message.data) as MatchingRealtimeEvent);
      } catch {
        // Ignore malformed frames (e.g. keepalive pings that aren't JSON).
      }
    };

    return () => socket.close();
  }, [userId]);

  return status;
}
