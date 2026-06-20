"use client";

import { useEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { BASE_PATH } from "../../site.config";

const WALK_SRC = `${BASE_PATH}/mascot/olm-mascot-walk.gif`;
const CELEBRATE_SRC = `${BASE_PATH}/mascot/olm-mascot-celebrate.png`;
const WALK_INTERVAL_MS = 5 * 60 * 1000;
const WALK_DURATION_MS = 14 * 1000;
const CELEBRATION_MS = 1800;
const TEST_WALK_PARAM = "mascotWalk";

const CONFETTI = [
  { dx: -34, dy: -58, color: "#b30000" },
  { dx: 24, dy: -64, color: "#111111" },
  { dx: -16, dy: -78, color: "#f4f4f0" },
  { dx: 42, dy: -36, color: "#b30000" },
  { dx: -44, dy: -28, color: "#111111" },
];

type WalkState = {
  id: number;
  direction: "ltr" | "rtl";
};

type CelebrationState = {
  id: number;
  x: number;
  y: number;
};

function reducedMotionPreferred() {
  return (
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function copyPosition() {
  const fallback = {
    x: window.innerWidth - 92,
    y: window.innerHeight - 112,
  };
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0) return fallback;

  const rect = selection.getRangeAt(0).getBoundingClientRect();
  if (!rect.width && !rect.height) return fallback;

  return {
    x: clamp(rect.left + rect.width / 2, 76, window.innerWidth - 76),
    y: clamp(rect.top, 112, window.innerHeight - 48),
  };
}

export default function MascotLayer() {
  const [walk, setWalk] = useState<WalkState | null>(null);
  const [celebrations, setCelebrations] = useState<CelebrationState[]>([]);
  const copyCounterRef = useRef(0);

  useEffect(() => {
    if (reducedMotionPreferred()) return undefined;

    let closed = false;
    const timers: number[] = [];

    const startWalk = () => {
      const id = Date.now();
      setWalk({
        id,
        direction: Math.random() < 0.5 ? "ltr" : "rtl",
      });
      timers.push(window.setTimeout(() => setWalk(null), WALK_DURATION_MS));
    };

    const scheduleWalk = () => {
      const timer = window.setTimeout(() => {
        if (closed) return;
        startWalk();
        scheduleWalk();
      }, WALK_INTERVAL_MS);
      timers.push(timer);
    };

    if (new URLSearchParams(window.location.search).has(TEST_WALK_PARAM)) {
      startWalk();
    }
    scheduleWalk();

    return () => {
      closed = true;
      timers.forEach((timer) => window.clearTimeout(timer));
    };
  }, []);

  useEffect(() => {
    if (reducedMotionPreferred()) return undefined;

    const onCopy = () => {
      copyCounterRef.current += 1;
      if (copyCounterRef.current % 3 !== 0) return;

      const id = Date.now();
      const position = copyPosition();
      setCelebrations((current) => [...current.slice(-1), { id, ...position }]);
      window.setTimeout(() => {
        setCelebrations((current) => current.filter((item) => item.id !== id));
      }, CELEBRATION_MS);
    };

    window.addEventListener("copy", onCopy);
    return () => window.removeEventListener("copy", onCopy);
  }, []);

  return (
    <div className="mascot-layer" aria-hidden="true">
      {walk && (
        <div
          key={walk.id}
          className={`mascot-walk mascot-walk-${walk.direction}`}
        >
          <img src={WALK_SRC} alt="" draggable={false} />
        </div>
      )}

      {celebrations.map((item) => (
        <div
          key={item.id}
          className="mascot-celebration"
          style={
            {
              "--mascot-x": `${item.x}px`,
              "--mascot-y": `${item.y}px`,
            } as CSSProperties
          }
        >
          <img src={CELEBRATE_SRC} alt="" draggable={false} />
          {CONFETTI.map((piece, index) => (
            <span
              key={index}
              className="mascot-confetti"
              style={
                {
                  "--confetti-x": `${piece.dx}px`,
                  "--confetti-y": `${piece.dy}px`,
                  backgroundColor: piece.color,
                } as CSSProperties
              }
            />
          ))}
        </div>
      ))}
    </div>
  );
}
