"use client";

import { useRef, useEffect, useCallback } from "react";

const CHARS = "!<>-_\\/[]{}—=+*^?#________";

function randomChar() {
  return CHARS[Math.floor(Math.random() * CHARS.length)];
}

interface Props {
  text: string;
  scrambleOnMount?: boolean;
  className?: string;
}

export default function ScrambleText({
  text,
  scrambleOnMount = false,
  className,
}: Props) {
  const elRef = useRef<HTMLSpanElement>(null);
  const frameRef = useRef<number>(0);

  const scramble = useCallback((targetText: string) => {
    const el = elRef.current;
    if (!el) return;

    const oldText = el.innerText;
    const length = Math.max(oldText.length, targetText.length);

    type Item = { from: string; to: string; start: number; end: number; char?: string };
    const queue: Item[] = [];

    for (let i = 0; i < length; i++) {
      const from = oldText[i] || "";
      const to = targetText[i] || "";
      const start = Math.floor(Math.random() * 40);
      const end = start + Math.floor(Math.random() * 40);
      queue.push({ from, to, start, end });
    }

    cancelAnimationFrame(frameRef.current);
    let frame = 0;

    const update = () => {
      let output = "";
      let complete = 0;

      for (const item of queue) {
        if (frame >= item.end) {
          complete++;
          output += item.to === "\n" ? "<br>" : item.to;
        } else if (frame >= item.start) {
          if (!item.char || Math.random() < 0.28) item.char = randomChar();
          output += `<span style="color:var(--text-muted)">${item.char}</span>`;
        } else {
          output += item.from === "\n" ? "<br>" : item.from;
        }
      }

      el.innerHTML = output;

      if (complete < queue.length) {
        frameRef.current = requestAnimationFrame(update);
        frame++;
      }
    };

    update();
  }, []);

  useEffect(() => {
    if (scrambleOnMount) scramble(text);
    return () => cancelAnimationFrame(frameRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <span ref={elRef} className={className} onMouseEnter={() => scramble(text)}>
      {text}
    </span>
  );
}
