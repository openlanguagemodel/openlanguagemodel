/* Typed access to the single-GPU throughput catalog for the /benchmarks page.
 *
 * The data comes from benchmarks.generated.json, written at build time by
 * benchmarks/throughput/build_site_data.py from the committed catalog CSV and
 * price snapshots. Nothing here is hand-typed: the Python side has already
 * chosen one batch per (GPU, model, context) cell, and these helpers only
 * look cells up, price them and format them.
 */
import raw from "./benchmarks.generated.json";

export type BatchResult = {
  batch: number;
  seeds: number[];
  tps: number;
  stepMs: number;
  mfu: number;
  mfuBoard: number;
  vramGb: number;
  vramPct: number;
  reservedGb: number;
};

export type Cell = {
  gpu: string;
  model: string;
  context: number;
  status: "measured" | "oom";
  seedsRun: number[];
  selectedBatch: number | null;
  batches: BatchResult[];
  oom: { batch: number; seeds: number[] }[];
  buildOom: boolean;
  edgeBatches: number[];
  searchCapped: boolean;
};

export type Gpu = {
  id: string;
  name: string;
  vramGb: number | null;
  ampDtype: string;
  peakTflops: number;
  boardPeakTflops: number;
};

export type PendingGpu = {
  id: string;
  name: string;
  vramGb: number | null;
  status: "not_yet_measured" | "not_in_catalog";
  note: string;
};

export type Quote = {
  usd: number;
  product: string;
  priceType: string;
  match: string;
  gpusPerInstance: number;
  notes: string;
  url: string;
};

export type Provider = {
  id: string;
  label: string;
  measuredHost: boolean;
  capturedAt: string;
  quotes: Record<string, Quote>;
};

export type ScalingRow = {
  gpus: number;
  replicates: number;
  tps: number;
  tpsStd: number;
  efficiency: number;
  peakGb: number;
};

export type BenchmarkData = {
  schemaVersion: number;
  source: {
    generator: string;
    catalog: string;
    catalogSha256: string;
    catalogRows: number;
    runpodPricing: { capturedAt: string; url: string; notes: string[] };
    providerPricing: {
      capturedAt: string;
      policy: { benchmark_transfer: string; excluded_costs: string[]; warning: string };
    };
  };
  protocolSeeds: number[];
  models: { key: string; label: string; parameters: number }[];
  contexts: number[];
  gpus: Gpu[];
  pendingGpus: PendingGpu[];
  cells: Cell[];
  providers: Provider[];
  scaling: {
    workload: Record<string, string | number | boolean>;
    source: { paper: string; repository_branch: string; commit: string };
    rows: ScalingRow[];
  };
};

export const DATA = raw as unknown as BenchmarkData;

const CELLS = new Map(DATA.cells.map((cell) => [cellKey(cell.gpu, cell.model, cell.context), cell]));

function cellKey(gpu: string, model: string, context: number): string {
  return `${gpu}|${model}|${context}`;
}

export function cellFor(gpu: string, model: string, context: number): Cell | undefined {
  return CELLS.get(cellKey(gpu, model, context));
}

export function selectedBatch(cell: Cell | undefined): BatchResult | undefined {
  if (!cell || cell.selectedBatch === null) return undefined;
  return cell.batches.find((item) => item.batch === cell.selectedBatch);
}

/** Why a (GPU, model, context) cell has no rows at all. */
export function notRunReason(gpu: string, model: string, context: number): string {
  const baseline = cellFor(gpu, model, 2048);
  if (!selectedBatch(baseline)) {
    return "Not run: the model does not fit at 2,048 tokens, so the context sweep skips it.";
  }
  const previous = DATA.contexts.filter((value) => value < context).pop();
  if (context > 8192 && previous && !selectedBatch(cellFor(gpu, model, previous))) {
    return `Not run: out of memory at ${previous.toLocaleString("en-US")} tokens.`;
  }
  return "Not run.";
}

export function usdPerBillionTokens(tokensPerSecond: number, usdPerHour: number): number {
  return (1e9 / tokensPerSecond / 3600) * usdPerHour;
}

export function providerById(id: string): Provider {
  return DATA.providers.find((provider) => provider.id === id) ?? DATA.providers[0];
}

export function modelLabel(key: string): string {
  return DATA.models.find((model) => model.key === key)?.label ?? key;
}

export function gpuLabel(gpu: Gpu | PendingGpu): string {
  return gpu.vramGb !== null && !gpu.name.includes("GB")
    ? `${gpu.name} ${gpu.vramGb} GB`
    : gpu.name;
}

/* ---------- formatting ---------- */

export function formatInteger(value: number): string {
  return Math.round(value).toLocaleString("en-US");
}

export function formatCompact(value: number): string {
  if (value >= 1e9) return `${trim(value / 1e9)}B`;
  if (value >= 1e6) return `${trim(value / 1e6)}M`;
  if (value >= 1e3) return `${trim(value / 1e3)}k`;
  return formatInteger(value);
}

/** Three significant digits without trailing zeros: 1, 2.5, 58.1, 140. */
function trim(value: number): string {
  return String(Number(value.toPrecision(3)));
}

export function formatUsd(value: number): string {
  if (value >= 1000) return `$${Math.round(value).toLocaleString("en-US")}`;
  if (value >= 100) return `$${value.toFixed(0)}`;
  return `$${value.toFixed(2)}`;
}

export function formatHours(hours: number): string {
  if (hours < 1) return `${Math.max(1, Math.round(hours * 60))} min`;
  if (hours < 48) return `${hours.toFixed(hours < 10 ? 1 : 0)} h`;
  const days = hours / 24;
  return `${days.toFixed(days < 10 ? 1 : 0)} days`;
}

export function formatPct(value: number): string {
  return `${value.toFixed(1)}%`;
}

export function formatGb(value: number): string {
  return `${value.toFixed(1)} GB`;
}

export function formatDate(timestamp: string): string {
  return new Date(timestamp).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}

export function formatContext(context: number): string {
  return context >= 1024 ? `${context / 1024}K` : String(context);
}

export const PRICE_TYPE_LABELS: Record<string, string> = {
  on_demand: "on-demand",
  marketplace_on_demand: "marketplace",
  capacity_block: "capacity block",
  flex_start: "flex start",
};
