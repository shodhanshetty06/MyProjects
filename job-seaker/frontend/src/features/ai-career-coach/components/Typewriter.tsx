import { useEffect, useState } from "react";

/**
 * Client-side reveal animation for chat responses. The backend returns the
 * full assistant message in one round trip (no token streaming endpoint
 * exists) - this simulates the feel by revealing it progressively.
 */
export function Typewriter({ text, onDone, speedMsPerChar = 8 }: { text: string; onDone?: () => void; speedMsPerChar?: number }) {
  const [shown, setShown] = useState("");

  useEffect(() => {
    setShown("");
    if (!text) return;
    let i = 0;
    const step = Math.max(1, Math.round(text.length / 120));
    const interval = window.setInterval(() => {
      i += step;
      setShown(text.slice(0, i));
      if (i >= text.length) {
        window.clearInterval(interval);
        onDone?.();
      }
    }, speedMsPerChar);
    return () => window.clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text]);

  return <>{shown}</>;
}
