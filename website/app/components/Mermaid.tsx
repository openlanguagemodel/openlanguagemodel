"use client";

import { useEffect } from "react";

/** Renders any <div class="mermaid"> nodes on the page after mount. */
export default function Mermaid() {
  useEffect(() => {
    let cancelled = false;
    const nodes = Array.from(
      document.querySelectorAll<HTMLElement>(".mermaid")
    );
    if (!nodes.length) return;

    import("mermaid")
      .then(({ default: mermaid }) => {
        if (cancelled) return;
        mermaid.initialize({
          startOnLoad: false,
          theme: "neutral",
          securityLevel: "loose",
          fontFamily: "var(--font-mono)",
        });
        return mermaid.run({ nodes });
      })
      .catch(() => {
        /* mermaid is optional; ignore render failures */
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return null;
}
