"use client";

/* Cost against wall-clock time for one training run, one point per GPU.
 *
 * Both axes are logarithmic because GPUs differ by several times in speed and
 * price. Points on the cost-time frontier (no other GPU is both faster and
 * cheaper) are drawn in the accent colour and joined by a line; every point is
 * labelled with its GPU and has a hover and keyboard tooltip. The table next
 * to the chart carries the same numbers for readers who cannot use it.
 */
import { useEffect, useRef, useState } from "react";
import { formatCompact, formatHours, formatInteger, formatUsd } from "../lib/benchmarks";

export type ChartPoint = {
  id: string;
  name: string;
  label: string;
  hours: number;
  cost: number;
  price: number;
  tokensPerSecond: number;
  batch: number;
  pareto: boolean;
};

const CHART_HEIGHT = 320;
const MARGIN = { top: 18, right: 20, bottom: 48, left: 64 };

export default function CostTimeChart({
  points: allPoints,
  tokenCount,
}: {
  points: ChartPoint[];
  tokenCount: number;
}) {
  // Log axes need finite, positive values; anything else is left off the chart
  // (the table still shows it).
  const points = allPoints.filter(
    (point) => plottable(point.hours) && plottable(point.cost)
  );
  const container = useRef<HTMLElement>(null);
  const [width, setWidth] = useState(720);
  const [active, setActive] = useState<string | null>(null);
  const empty = points.length === 0;

  useEffect(() => {
    const node = container.current;
    if (!node) return;
    const observer = new ResizeObserver((entries) => {
      setWidth(Math.max(280, Math.floor(entries[0].contentRect.width)));
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [empty]);

  if (empty) return null;

  const hours = points.map((point) => point.hours);
  const costs = points.map((point) => point.cost);
  // Label time in days only when the whole axis is past two days.
  const days = Math.min(...hours) >= 48;
  const x = logScale(hours, MARGIN.left, width - MARGIN.right, days ? 24 : 1);
  const y = logScale(costs, CHART_HEIGHT - MARGIN.bottom, MARGIN.top);
  const frontier = points
    .filter((point) => point.pareto)
    .sort((a, b) => a.hours - b.hours);
  const labels = placeLabels(points, x, y, width);
  const hovered = points.find((point) => point.id === active);

  return (
    <figure className="bench-chart" ref={container}>
      <figcaption>
        <span>
          Time and compute cost for {formatCompact(tokenCount)} tokens, one GPU.
          Both axes are logarithmic.
        </span>
        <span className="bench-chart-legend" aria-hidden="true">
          <span className="key pareto" /> on the cost–time frontier
          <span className="key other" /> slower and costlier than another GPU
        </span>
      </figcaption>
      <svg
        width={width}
        height={CHART_HEIGHT}
        viewBox={`0 0 ${width} ${CHART_HEIGHT}`}
        role="group"
        aria-label="Cost against training time for each measured GPU"
      >
        {x.ticks.map((tick) => (
          <g key={`x${tick}`}>
            <line
              className="grid"
              x1={x.scale(tick)}
              x2={x.scale(tick)}
              y1={MARGIN.top}
              y2={CHART_HEIGHT - MARGIN.bottom}
            />
            <text
              className="tick"
              x={x.scale(tick)}
              y={CHART_HEIGHT - MARGIN.bottom + 18}
              textAnchor="middle"
            >
              {days ? `${Number((tick / 24).toPrecision(3))} d` : formatHoursTick(tick)}
            </text>
          </g>
        ))}
        {y.ticks.map((tick) => (
          <g key={`y${tick}`}>
            <line
              className="grid"
              x1={MARGIN.left}
              x2={width - MARGIN.right}
              y1={y.scale(tick)}
              y2={y.scale(tick)}
            />
            <text className="tick" x={MARGIN.left - 8} y={y.scale(tick) + 4} textAnchor="end">
              {Number.isInteger(tick) ? `$${tick.toLocaleString("en-US")}` : formatUsd(tick)}
            </text>
          </g>
        ))}
        <text
          className="axis-title"
          x={(MARGIN.left + width - MARGIN.right) / 2}
          y={CHART_HEIGHT - 8}
          textAnchor="middle"
        >
          Wall-clock time
        </text>
        <text
          className="axis-title"
          transform={`translate(14 ${(MARGIN.top + CHART_HEIGHT - MARGIN.bottom) / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          Compute cost
        </text>
        {frontier.length > 1 && (
          <polyline
            className="frontier"
            points={frontier.map((point) => `${x.scale(point.hours)},${y.scale(point.cost)}`).join(" ")}
          />
        )}
        {labels.map((label) => (
          <text
            key={`label-${label.point.id}`}
            aria-hidden="true"
            className={label.point.pareto ? "point-label pareto" : "point-label"}
            x={label.x}
            y={label.y}
            textAnchor={label.anchor}
          >
            {label.text}
          </text>
        ))}
        {points.map((point) => (
          <g
            key={point.id}
            className={point.pareto ? "point pareto" : "point"}
            tabIndex={0}
            role="img"
            aria-label={`${point.name}: ${formatHours(point.hours)}, ${formatUsd(point.cost)}, ${formatInteger(point.tokensPerSecond)} tokens/s at batch ${point.batch}${point.pareto ? ", on the frontier" : ""}`}
            onPointerEnter={() => setActive(point.id)}
            onPointerLeave={() => setActive(null)}
            onFocus={() => setActive(point.id)}
            onBlur={() => setActive(null)}
          >
            <circle className="hit" cx={x.scale(point.hours)} cy={y.scale(point.cost)} r={14} />
            <circle className="dot" cx={x.scale(point.hours)} cy={y.scale(point.cost)} r={5.5} />
          </g>
        ))}
      </svg>
      {hovered && (
        <div
          className="bench-tooltip"
          aria-hidden="true"
          style={{
            left: Math.min(x.scale(hovered.hours) + 12, width - 190),
            top: Math.max(0, y.scale(hovered.cost) - 70),
          }}
        >
          <strong>{formatUsd(hovered.cost)}</strong> · {formatHours(hovered.hours)}
          <span>{hovered.label}</span>
          <span>
            {formatInteger(hovered.tokensPerSecond)} tokens/s · batch {hovered.batch}
          </span>
          <span>{formatUsd(hovered.price)}/GPU-hour</span>
        </div>
      )}
    </figure>
  );
}

type Scale = { scale: (value: number) => number; ticks: number[] };

function plottable(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}

function formatHoursTick(hours: number): string {
  return hours < 1
    ? `${Number((hours * 60).toPrecision(3))} min`
    : `${Number(hours.toPrecision(3))} h`;
}

/* Tick mantissas from coarse to fine; the first set that puts at least three
   ticks inside the domain is used, so narrow ranges still get a readable axis. */
const TICK_STEPS = [
  [1, 2, 5],
  [1, 1.5, 2, 3, 5, 7],
  [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8],
];

/**
 * A log10 scale padded around the data, with round ticks inside it. Ticks are
 * round in `unit` (24 puts time ticks on whole or half days).
 */
function logScale(values: number[], from: number, to: number, unit = 1): Scale {
  let low = Math.min(...values);
  let high = Math.max(...values);
  // log10 of zero or infinity would make the tick loop below never end.
  if (!(low > 0 && Number.isFinite(high))) return { scale: () => from, ticks: [] };
  if (high / low < 2.5) {
    const middle = Math.sqrt(low * high);
    low = middle / 1.6;
    high = middle * 1.6;
  } else {
    low /= 1.25;
    high *= 1.25;
  }
  const span = Math.log10(high) - Math.log10(low);
  const scale = (value: number) => from + ((Math.log10(value) - Math.log10(low)) / span) * (to - from);
  let ticks: number[] = [];
  for (const steps of TICK_STEPS) {
    ticks = [];
    const [lowUnits, highUnits] = [low / unit, high / unit];
    for (
      let exponent = Math.floor(Math.log10(lowUnits));
      exponent <= Math.ceil(Math.log10(highUnits));
      exponent++
    ) {
      for (const step of steps) {
        const tick = Number((step * 10 ** exponent).toPrecision(3));
        if (tick >= lowUnits && tick <= highUnits) ticks.push(tick * unit);
      }
    }
    if (ticks.length >= 3) break;
  }
  return { scale, ticks };
}

type Label = {
  point: ChartPoint;
  text: string;
  x: number;
  y: number;
  anchor: "start" | "middle" | "end";
};

type Box = { left: number; right: number; top: number; bottom: number };

/** "H100 SXM 80GB" -> "H100", "RTX PRO 6000" stays: names for narrow charts. */
function shortName(name: string): string {
  return name.replace(/\s+\d+\s?GB$/, "").replace(/\s+(SXM|PCIe)$/, "");
}

/**
 * Put each GPU name next to its point: to the right if it fits, otherwise to
 * the left, above or below, skipping spots that cover another label or dot.
 * Frontier GPUs are placed first so they get the best spots.
 */
function placeLabels(points: ChartPoint[], x: Scale, y: Scale, width: number): Label[] {
  const narrow = width < 560;
  const charWidth = 6.4;
  const plot: Box = {
    left: MARGIN.left,
    right: width - 4,
    top: 0,
    bottom: CHART_HEIGHT - MARGIN.bottom,
  };
  const dots = points.map((point) => ({ x: x.scale(point.hours), y: y.scale(point.cost) }));
  const placed: Box[] = [];
  const order = [...points].sort((a, b) => Number(b.pareto) - Number(a.pareto));

  return order.map((point) => {
    const text = narrow ? shortName(point.name) : point.name;
    const textWidth = text.length * charWidth;
    const px = x.scale(point.hours);
    const py = y.scale(point.cost);
    const candidates: Omit<Label, "point" | "text">[] = [
      { x: px + 9, y: py + 4, anchor: "start" },
      { x: px - 9, y: py + 4, anchor: "end" },
      { x: px, y: py - 11, anchor: "middle" },
      { x: px, y: py + 20, anchor: "middle" },
    ];
    const boxOf = (candidate: Omit<Label, "point" | "text">): Box => {
      const left =
        candidate.anchor === "start"
          ? candidate.x
          : candidate.anchor === "end"
            ? candidate.x - textWidth
            : candidate.x - textWidth / 2;
      return { left, right: left + textWidth, top: candidate.y - 10, bottom: candidate.y + 3 };
    };
    const fits = (box: Box) =>
      box.left >= plot.left &&
      box.right <= plot.right &&
      box.top >= plot.top &&
      box.bottom <= plot.bottom &&
      !placed.some(
        (other) =>
          box.left < other.right &&
          box.right > other.left &&
          box.top < other.bottom &&
          box.bottom > other.top
      ) &&
      !dots.some(
        (dot) =>
          dot.x > box.left - 6 &&
          dot.x < box.right + 6 &&
          dot.y > box.top - 6 &&
          dot.y < box.bottom + 6
      );
    const choice =
      candidates.find((candidate) => fits(boxOf(candidate))) ??
      candidates[px + 9 + textWidth > plot.right ? 1 : 0];
    placed.push(boxOf(choice));
    return { point, text, ...choice };
  });
}
