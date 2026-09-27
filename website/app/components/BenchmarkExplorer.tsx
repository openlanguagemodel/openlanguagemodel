"use client";

import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import {
  DATA,
  PRICE_TYPE_LABELS,
  cellFor,
  formatCompact,
  formatContext,
  formatDate,
  formatGb,
  formatHours,
  formatInteger,
  formatPct,
  formatUsd,
  gpuLabel,
  modelLabel,
  notRunReason,
  providerById,
  selectedBatch,
  type BatchResult,
  type Cell,
  type Gpu,
  type Quote,
} from "../lib/benchmarks";

/* One row of the comparison: a GPU for the chosen model, context and price. */
type Row = {
  gpu: Gpu;
  cell: Cell | undefined;
  result: BatchResult | undefined;
  quote: Quote | undefined;
  price: number | null;
  overridden: boolean;
  hours: number | null;
  cost: number | null;
  fastest: boolean;
  cheapest: boolean;
  pareto: boolean;
};

const DEFAULT_STATE = {
  model: "500m",
  context: 2048,
  tokens: "1",
  unit: "B" as "M" | "B",
  provider: "runpod-secure",
  sort: "cost" as "cost" | "time",
};
const UNIT_SCALE = { M: 1e6, B: 1e9 };
const PRICE_STORAGE_KEY = "olm-benchmarks-price-overrides-v1";

export default function BenchmarkExplorer() {
  const [model, setModel] = useState(DEFAULT_STATE.model);
  const [context, setContext] = useState(DEFAULT_STATE.context);
  const [tokens, setTokens] = useState(DEFAULT_STATE.tokens);
  const [unit, setUnit] = useState(DEFAULT_STATE.unit);
  const [providerId, setProviderId] = useState(DEFAULT_STATE.provider);
  const [sort, setSort] = useState(DEFAULT_STATE.sort);
  const [editingPrices, setEditingPrices] = useState(false);
  const [overrides, setOverrides] = useState<Record<string, number>>({});
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  // Becomes true after the URL and saved prices are read, so the effects that
  // write them back never run with the server-rendered defaults.
  const [hydrated, setHydrated] = useState(false);

  // Read a shared view from the URL and any saved price edits, once.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const queryModel = params.get("model");
    if (queryModel && DATA.models.some((item) => item.key === queryModel)) {
      setModel(queryModel);
    }
    const queryContext = Number(params.get("context"));
    if (DATA.contexts.includes(queryContext)) setContext(queryContext);
    const queryTokens = Number(params.get("tokens"));
    if (Number.isFinite(queryTokens) && queryTokens > 0) {
      const inBillions = queryTokens >= 1e9;
      setUnit(inBillions ? "B" : "M");
      setTokens(String(queryTokens / (inBillions ? 1e9 : 1e6)));
    }
    const queryProvider = params.get("provider");
    if (queryProvider && DATA.providers.some((item) => item.id === queryProvider)) {
      setProviderId(queryProvider);
    }
    try {
      const saved = JSON.parse(window.localStorage.getItem(PRICE_STORAGE_KEY) ?? "{}");
      if (saved && typeof saved === "object") setOverrides(saved);
    } catch {
      // Storage can be unavailable (private windows); edits then last one visit.
    }
    setHydrated(true);
  }, []);

  const tokenCount = Math.max(0, Number(tokens) || 0) * UNIT_SCALE[unit];

  useEffect(() => {
    if (!hydrated) return;
    const params = new URLSearchParams({
      model,
      context: String(context),
      tokens: String(tokenCount),
      provider: providerId,
    });
    window.history.replaceState(null, "", `${window.location.pathname}?${params}`);
  }, [hydrated, model, context, tokenCount, providerId]);

  useEffect(() => {
    if (!hydrated) return;
    try {
      window.localStorage.setItem(PRICE_STORAGE_KEY, JSON.stringify(overrides));
    } catch {
      // Ignore: the override still applies for this visit.
    }
  }, [hydrated, overrides]);

  const provider = providerById(providerId);
  const rows = useMemo(
    () => buildRows(model, context, tokenCount, providerId, overrides),
    [model, context, tokenCount, providerId, overrides]
  );
  const ordered = useMemo(() => sortRows(rows, sort), [rows, sort]);
  const priced = rows.filter((row) => row.cost !== null);
  const fastest = rows.find((row) => row.fastest);
  const cheapest = rows.find((row) => row.cheapest);
  const parameters = DATA.models.find((item) => item.key === model)?.parameters ?? 0;
  const tableCaption =
    `Best measured batch per GPU for ${modelLabel(model)} at ` +
    `${context.toLocaleString("en-US")} tokens; time and cost for ` +
    `${formatCompact(tokenCount)} tokens.`;

  const setOverride = (gpu: string, value: string) => {
    const key = `${providerId}|${gpu}`;
    const price = Number(value);
    setOverrides((current) => {
      const next = { ...current };
      if (value === "" || !Number.isFinite(price) || price < 0) delete next[key];
      else next[key] = price;
      return next;
    });
  };

  return (
    <div className="bench-explorer">
      <form className="bench-controls" onSubmit={(event) => event.preventDefault()}>
        <fieldset className="bench-field">
          <legend>Model size</legend>
          <div className="bench-choices">
            {DATA.models.map((item) => (
              <label className="bench-choice" key={item.key}>
                <input
                  type="radio"
                  name="bench-model"
                  value={item.key}
                  checked={model === item.key}
                  onChange={() => setModel(item.key)}
                />
                <span>{item.label}</span>
              </label>
            ))}
          </div>
        </fieldset>

        <fieldset className="bench-field">
          <legend>Context length (tokens)</legend>
          <div className="bench-choices">
            {DATA.contexts.map((value) => (
              <label className="bench-choice" key={value}>
                <input
                  type="radio"
                  name="bench-context"
                  value={value}
                  checked={context === value}
                  onChange={() => setContext(value)}
                />
                <span>{formatContext(value)}</span>
              </label>
            ))}
          </div>
        </fieldset>

        <div className="bench-field-row">
          <fieldset className="bench-field">
            <legend>Training tokens</legend>
            <div className="bench-inline">
              <input
                className="bench-input"
                type="number"
                min="0"
                step="any"
                inputMode="decimal"
                value={tokens}
                aria-label="Number of training tokens"
                onChange={(event) => setTokens(event.target.value)}
              />
              <select
                className="bench-select"
                value={unit}
                aria-label="Token unit"
                onChange={(event) => setUnit(event.target.value as "M" | "B")}
              >
                <option value="M">million</option>
                <option value="B">billion</option>
              </select>
              <button
                type="button"
                className="bench-button quiet"
                title="About 20 training tokens per parameter, the compute-optimal ratio reported by Hoffmann et al. (2022)"
                onClick={() => {
                  const total = 20 * parameters;
                  const inBillions = total >= 1e9;
                  setUnit(inBillions ? "B" : "M");
                  setTokens(String(+(total / (inBillions ? 1e9 : 1e6)).toPrecision(3)));
                }}
              >
                20 tokens / parameter
              </button>
            </div>
          </fieldset>

          <fieldset className="bench-field">
            <legend>Price</legend>
            <div className="bench-inline">
              <select
                className="bench-select"
                value={providerId}
                aria-label="Price source"
                onChange={(event) => setProviderId(event.target.value)}
              >
                {DATA.providers.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.label}
                  </option>
                ))}
              </select>
              <span className="bench-muted">quoted {formatDate(provider.capturedAt)}</span>
            </div>
          </fieldset>
        </div>
      </form>

      <p className="bench-context-note">
        {modelLabel(model)} parameters ({formatInteger(parameters)}),{" "}
        {context.toLocaleString("en-US")}-token context,{" "}
        {tokenCount > 0 ? `${formatCompact(tokenCount)} tokens` : "no tokens"} at{" "}
        {provider.label} prices.
        {!provider.measuredHost && (
          <>
            {" "}
            <strong>Projection:</strong> tokens/s were measured on RunPod and
            applied to {provider.label}&rsquo;s quoted rates.
          </>
        )}
      </p>

      <div className="bench-summary" aria-live="polite">
        <SummaryItem
          label="Fastest"
          row={fastest}
          value={fastest?.hours != null ? formatHours(fastest.hours) : null}
        />
        <SummaryItem
          label="Cheapest"
          row={cheapest}
          value={cheapest?.cost != null ? formatUsd(cheapest.cost) : null}
        />
        <div className="bench-summary-item">
          <span className="bench-summary-label">Cost–time frontier</span>
          <span className="bench-summary-value">
            {priced.length
              ? rows
                  .filter((row) => row.pareto)
                  .map((row) => row.gpu.name)
                  .join(" · ")
              : "No priced GPU fits"}
          </span>
        </div>
      </div>

      {priced.length > 0 && <CostTimeChart rows={priced} tokenCount={tokenCount} />}

      <div className="bench-table-tools">
        <label className="bench-inline bench-muted">
          Sort
          <select
            className="bench-select"
            value={sort}
            onChange={(event) => setSort(event.target.value as "cost" | "time")}
          >
            <option value="cost">cheapest first</option>
            <option value="time">fastest first</option>
          </select>
        </label>
        <button
          type="button"
          className="bench-button"
          aria-pressed={editingPrices}
          onClick={() => setEditingPrices((value) => !value)}
        >
          {editingPrices ? "Done editing prices" : "Edit prices"}
        </button>
        {Object.keys(overrides).length > 0 && (
          <button type="button" className="bench-button quiet" onClick={() => setOverrides({})}>
            Reset prices
          </button>
        )}
        <button
          type="button"
          className="bench-button"
          onClick={() => downloadCsv(rows, model, context, tokenCount, provider.label)}
        >
          Download CSV
        </button>
      </div>

      {/* The caption is repeated outside the scroller so a narrow screen shows
          all of it; screen readers get the real <caption>. */}
      <p className="bench-caption" aria-hidden="true">
        {tableCaption}
      </p>
      <div
        className="bench-table-wrap"
        role="region"
        aria-label="GPU comparison table"
        tabIndex={0}
      >
        <table className="bench-table">
          <caption className="sr-only">{tableCaption}</caption>
          <thead>
            <tr>
              <th scope="col">GPU</th>
              <th scope="col" className="num">Batch</th>
              <th scope="col" className="num">Tokens/s</th>
              <th scope="col" className="num">MFU</th>
              <th scope="col" className="num">Peak VRAM</th>
              <th scope="col" className="num">$/GPU-hour</th>
              <th scope="col" className="num">Time</th>
              <th scope="col" className="num">Cost</th>
              <th scope="col">Seeds</th>
            </tr>
          </thead>
          <tbody>
            {ordered.map((row) => {
              const open = Boolean(expanded[row.gpu.id]);
              const detailId = `bench-detail-${slug(row.gpu.id)}`;
              return (
                <Fragment key={row.gpu.id}>
                  <tr className={rowClass(row)}>
                    <th scope="row">
                      <span className="bench-gpu">{gpuLabel(row.gpu)}</span>
                      <Badges row={row} />
                      {row.cell && (
                        <button
                          type="button"
                          className="bench-disclosure"
                          aria-expanded={open}
                          aria-controls={detailId}
                          onClick={() =>
                            setExpanded((current) => ({ ...current, [row.gpu.id]: !open }))
                          }
                        >
                          {open ? "Hide batch sweep" : "Batch sweep"}
                        </button>
                      )}
                    </th>
                    {row.result ? (
                      <>
                        <td className="num">
                          {row.result.batch}
                          {row.cell?.searchCapped && (
                            <sup title="Largest batch tried; a larger batch was not tested">†</sup>
                          )}
                        </td>
                        <td className="num">{formatInteger(row.result.tps)}</td>
                        <td className="num" title={`Board clock: ${formatPct(row.result.mfuBoard)}`}>
                          {formatPct(row.result.mfu)}
                        </td>
                        <td className="num">
                          {formatGb(row.result.vramGb)}
                          <span className="bench-sub">{row.result.vramPct.toFixed(0)}%</span>
                        </td>
                        <td className="num">
                          <PriceCell
                            row={row}
                            editing={editingPrices}
                            onChange={(value) => setOverride(row.gpu.id, value)}
                          />
                        </td>
                        <td className="num">{row.hours !== null ? formatHours(row.hours) : "—"}</td>
                        <td className="num bench-strong">
                          {row.cost !== null ? formatUsd(row.cost) : "—"}
                        </td>
                        <td>
                          <Coverage cell={row.cell!} result={row.result} />
                        </td>
                      </>
                    ) : (
                      <td colSpan={8} className="bench-status">
                        {statusText(row, model, context)}
                      </td>
                    )}
                  </tr>
                  {row.cell && (
                    <tr id={detailId} className="bench-detail" hidden={!open}>
                      <td colSpan={9}>
                        <BatchSweep cell={row.cell} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="bench-footnote">
        MFU is measured against the GPU&rsquo;s published dense Tensor Core peak
        at its reference clock; hover a value for MFU at the board&rsquo;s
        reported clock. Peak VRAM is allocated memory and its share of the card.
        † The fastest batch was also the largest one the sweep tried. Cost is
        steady-state compute only.
      </p>

      <PendingList />

      <CoverageMatrix model={model} context={context} onPick={setContext} />
    </div>
  );
}

function buildRows(
  model: string,
  context: number,
  tokenCount: number,
  providerId: string,
  overrides: Record<string, number>
): Row[] {
  const provider = providerById(providerId);
  const rows: Row[] = DATA.gpus.map((gpu) => {
    const cell = cellFor(gpu.id, model, context);
    const result = selectedBatch(cell);
    const quote = provider.quotes[gpu.id];
    const override = overrides[`${providerId}|${gpu.id}`];
    const price = override ?? quote?.usd ?? null;
    const hours = result && tokenCount > 0 ? tokenCount / result.tps / 3600 : null;
    const cost = hours !== null && price !== null ? hours * price : null;
    return {
      gpu,
      cell,
      result,
      quote,
      price,
      overridden: override !== undefined,
      hours,
      cost,
      fastest: false,
      cheapest: false,
      pareto: false,
    };
  });

  const timed = rows.filter((row) => row.hours !== null);
  const priced = rows.filter((row) => row.cost !== null);
  const minHours = Math.min(...timed.map((row) => row.hours!));
  const minCost = Math.min(...priced.map((row) => row.cost!));
  for (const row of timed) row.fastest = row.hours === minHours;
  for (const row of priced) {
    row.cheapest = row.cost === minCost;
    // On the frontier when no other GPU is both no slower and no costlier,
    // and strictly better in at least one of the two.
    row.pareto = !priced.some(
      (other) =>
        other !== row &&
        other.hours! <= row.hours! &&
        other.cost! <= row.cost! &&
        (other.hours! < row.hours! || other.cost! < row.cost!)
    );
  }
  return rows;
}

function sortRows(rows: Row[], sort: "cost" | "time"): Row[] {
  const rank = (row: Row) => (row.result ? (row.cost !== null ? 0 : 1) : row.cell ? 2 : 3);
  return [...rows].sort((a, b) => {
    const byRank = rank(a) - rank(b);
    if (byRank !== 0) return byRank;
    if (sort === "cost" && a.cost !== null && b.cost !== null) return a.cost - b.cost;
    if (a.hours !== null && b.hours !== null) return a.hours - b.hours;
    return 0;
  });
}

function rowClass(row: Row): string {
  if (!row.result) return "bench-row-missing";
  return row.pareto ? "bench-row-pareto" : "";
}

function statusText(row: Row, model: string, context: number): string {
  const cell = row.cell;
  if (!cell) return notRunReason(row.gpu.id, model, context);
  if (cell.buildOom) return "Out of memory while building the model: no batch was attempted.";
  const first = cell.oom[0];
  return first
    ? `Out of memory at every batch tried, down to batch ${first.batch}, for both seeds.`
    : "No complete batch.";
}

function slug(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-");
}

function Badges({ row }: { row: Row }) {
  const badges = [
    row.fastest && "Fastest",
    row.cheapest && "Cheapest",
    row.pareto && !row.fastest && !row.cheapest && "Frontier",
  ].filter(Boolean) as string[];
  if (!badges.length) return null;
  return (
    <span className="bench-badges">
      {badges.map((badge) => (
        <span className="bench-badge" key={badge}>
          {badge}
        </span>
      ))}
    </span>
  );
}

function PriceCell({
  row,
  editing,
  onChange,
}: {
  row: Row;
  editing: boolean;
  onChange: (value: string) => void;
}) {
  // While the field has focus it shows what was typed, so clearing it to type
  // a new price does not snap back to the quoted one.
  const [draft, setDraft] = useState<string | null>(null);
  if (editing) {
    return (
      <input
        className="bench-input bench-price-input"
        type="number"
        min="0"
        step="0.01"
        inputMode="decimal"
        aria-label={`Price per GPU-hour for ${row.gpu.name}`}
        value={draft ?? row.price ?? ""}
        placeholder="—"
        onChange={(event) => {
          setDraft(event.target.value);
          onChange(event.target.value);
        }}
        onBlur={() => setDraft(null)}
      />
    );
  }
  if (row.price === null) return <span className="bench-muted">no quote</span>;
  const quote = row.quote;
  const details = [
    row.overridden ? "your price" : null,
    !row.overridden && quote && quote.priceType !== "on_demand"
      ? PRICE_TYPE_LABELS[quote.priceType] ?? quote.priceType
      : null,
    !row.overridden && quote && quote.gpusPerInstance > 1
      ? `${quote.gpusPerInstance}-GPU instance`
      : null,
    !row.overridden && quote && quote.match !== "exact" ? "similar SKU" : null,
  ].filter(Boolean);
  return (
    <span title={!row.overridden && quote ? `${quote.product}. ${quote.notes}`.trim() : undefined}>
      {formatUsd(row.price)}
      {details.length > 0 && <span className="bench-sub">{details.join(" · ")}</span>}
    </span>
  );
}

function Coverage({ cell, result }: { cell: Cell; result: BatchResult }) {
  const all = DATA.protocolSeeds.length;
  const complete = result.seeds.length === all;
  return (
    <span className={complete ? "bench-seeds" : "bench-seeds partial"}>
      {result.seeds.length}/{all}
      {cell.edgeBatches.length > 0 && (
        <span className="bench-sub">
          batch {cell.edgeBatches.join(", ")} fit only some seeds
        </span>
      )}
    </span>
  );
}

function SummaryItem({
  label,
  row,
  value,
}: {
  label: string;
  row: Row | undefined;
  value: string | null;
}) {
  return (
    <div className="bench-summary-item">
      <span className="bench-summary-label">{label}</span>
      <span className="bench-summary-value">
        {row && value ? (
          <>
            <strong>{value}</strong> {row.gpu.name}
          </>
        ) : (
          "—"
        )}
      </span>
    </div>
  );
}

function BatchSweep({ cell }: { cell: Cell }) {
  const rows = [
    ...cell.batches.map((item) => ({ batch: item.batch, item })),
    ...cell.oom
      .filter((oom) => !cell.batches.some((item) => item.batch === oom.batch))
      .map((oom) => ({ batch: oom.batch, item: undefined })),
  ].sort((a, b) => a.batch - b.batch);
  return (
    <div className="bench-sweep">
      <p className="bench-muted">
        Every batch tried for this cell. Values are medians over the seeds that
        completed the batch.
        {cell.buildOom && " The model did not fit in memory before the first step."}
      </p>
      {rows.length > 0 && (
        <table className="bench-table bench-table-inner">
          <thead>
            <tr>
              <th scope="col" className="num">Batch</th>
              <th scope="col" className="num">Tokens/s</th>
              <th scope="col" className="num">Step</th>
              <th scope="col" className="num">MFU (spec)</th>
              <th scope="col" className="num">MFU (board)</th>
              <th scope="col" className="num">Peak VRAM</th>
              <th scope="col">Outcome</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ batch, item }) => {
              const oom = cell.oom.find((entry) => entry.batch === batch);
              const chosen = batch === cell.selectedBatch;
              return (
                <tr key={batch} className={chosen ? "bench-row-pareto" : undefined}>
                  <td className="num">{batch}</td>
                  <td className="num">{item ? formatInteger(item.tps) : "—"}</td>
                  <td className="num">{item ? `${item.stepMs.toFixed(0)} ms` : "—"}</td>
                  <td className="num">{item ? formatPct(item.mfu) : "—"}</td>
                  <td className="num">{item ? formatPct(item.mfuBoard) : "—"}</td>
                  <td className="num">{item ? formatGb(item.vramGb) : "—"}</td>
                  <td>
                    {chosen
                      ? "Selected"
                      : item && oom
                        ? `Fit seed ${item.seeds.join(", ")}; out of memory on seed ${oom.seeds.join(", ")}`
                        : item
                          ? `Completed (seeds ${item.seeds.join(", ")})`
                          : `Out of memory (seeds ${oom?.seeds.join(", ")})`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

function PendingList() {
  const notMeasured = DATA.pendingGpus.filter((gpu) => gpu.status === "not_yet_measured");
  const unlisted = DATA.pendingGpus.filter((gpu) => gpu.status === "not_in_catalog");
  if (!notMeasured.length && !unlisted.length) return null;
  return (
    <div className="bench-pending">
      {notMeasured.map((gpu) => (
        <p key={gpu.id}>
          <strong>{gpuLabel(gpu)}: not yet measured.</strong> {gpu.note}{" "}
          <a className="hover-link" href="#contribute">
            Contribute it →
          </a>
        </p>
      ))}
      {unlisted.length > 0 && (
        <p className="bench-muted">
          Priced but not in the catalog yet: {unlisted.map((gpu) => gpu.name).join(", ")}.
        </p>
      )}
    </div>
  );
}

function CoverageMatrix({
  model,
  context,
  onPick,
}: {
  model: string;
  context: number;
  onPick: (context: number) => void;
}) {
  return (
    <div className="bench-coverage">
      <h4>
        Where {modelLabel(model)} was measured
      </h4>
      <p className="bench-muted">
        Tokens/s of the best batch per GPU and context. Pick a cell to compare
        at that context.
      </p>
      <div
        className="bench-table-wrap"
        role="region"
        aria-label={`Coverage of ${modelLabel(model)} across GPUs and contexts`}
        tabIndex={0}
      >
        <table className="bench-matrix">
          <thead>
            <tr>
              <th scope="col">GPU</th>
              {DATA.contexts.map((value) => (
                <th
                  scope="col"
                  key={value}
                  className={value === context ? "current" : undefined}
                >
                  {formatContext(value)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {DATA.gpus.map((gpu) => (
              <tr key={gpu.id}>
                <th scope="row">{gpu.name}</th>
                {DATA.contexts.map((value) => {
                  const cell = cellFor(gpu.id, model, value);
                  const result = selectedBatch(cell);
                  const label = result
                    ? formatCompact(result.tps)
                    : cell
                      ? "OOM"
                      : "—";
                  const state = result ? "measured" : cell ? "oom" : "not-run";
                  const description = result
                    ? `${formatInteger(result.tps)} tokens/s at batch ${result.batch}`
                    : cell
                      ? "out of memory"
                      : "not run";
                  return (
                    <td
                      key={value}
                      className={`${state}${value === context ? " current" : ""}`}
                    >
                      <button
                        type="button"
                        onClick={() => onPick(value)}
                        aria-label={`${gpu.name}, ${value.toLocaleString("en-US")} tokens: ${description}`}
                        aria-pressed={value === context}
                      >
                        {label}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="bench-legend">
        <span className="measured">tokens/s</span> measured, best batch
        <span className="oom">OOM</span> out of memory at every batch tried
        <span className="not-run">—</span> not run
      </p>
    </div>
  );
}

/* ---------- cost vs time chart ---------- */

const CHART_HEIGHT = 320;
const MARGIN = { top: 18, right: 20, bottom: 48, left: 64 };

function CostTimeChart({ rows, tokenCount }: { rows: Row[]; tokenCount: number }) {
  const container = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(720);
  const [active, setActive] = useState<string | null>(null);

  useEffect(() => {
    const node = container.current;
    if (!node) return;
    const observer = new ResizeObserver((entries) => {
      setWidth(Math.max(280, Math.floor(entries[0].contentRect.width)));
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const hours = rows.map((row) => row.hours!);
  const costs = rows.map((row) => row.cost!);
  const x = logScale(hours, MARGIN.left, width - MARGIN.right);
  const y = logScale(costs, CHART_HEIGHT - MARGIN.bottom, MARGIN.top);
  const frontier = rows
    .filter((row) => row.pareto)
    .sort((a, b) => a.hours! - b.hours!);
  const labels = placeLabels(rows, x, y, width);
  const hovered = rows.find((row) => row.gpu.id === active);

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
              {formatHoursTick(tick)}
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
            points={frontier.map((row) => `${x.scale(row.hours!)},${y.scale(row.cost!)}`).join(" ")}
          />
        )}
        {labels.map((label) => (
          <text
            key={`label-${label.row.gpu.id}`}
            aria-hidden="true"
            className={label.row.pareto ? "point-label pareto" : "point-label"}
            x={label.x}
            y={label.y}
            textAnchor={label.anchor}
          >
            {label.text}
          </text>
        ))}
        {rows.map((row) => (
          <g
            key={row.gpu.id}
            className={row.pareto ? "point pareto" : "point"}
            tabIndex={0}
            role="img"
            aria-label={`${row.gpu.name}: ${formatHours(row.hours!)}, ${formatUsd(row.cost!)}, ${formatInteger(row.result!.tps)} tokens/s at batch ${row.result!.batch}${row.pareto ? ", on the frontier" : ""}`}
            onPointerEnter={() => setActive(row.gpu.id)}
            onPointerLeave={() => setActive(null)}
            onFocus={() => setActive(row.gpu.id)}
            onBlur={() => setActive(null)}
          >
            <circle className="hit" cx={x.scale(row.hours!)} cy={y.scale(row.cost!)} r={14} />
            <circle className="dot" cx={x.scale(row.hours!)} cy={y.scale(row.cost!)} r={5.5} />
          </g>
        ))}
      </svg>
      {hovered && (
        <div
          className="bench-tooltip"
          aria-hidden="true"
          style={{
            left: Math.min(x.scale(hovered.hours!) + 12, width - 190),
            top: Math.max(0, y.scale(hovered.cost!) - 70),
          }}
        >
          <strong>{formatUsd(hovered.cost!)}</strong> · {formatHours(hovered.hours!)}
          <span>{gpuLabel(hovered.gpu)}</span>
          <span>
            {formatInteger(hovered.result!.tps)} tokens/s · batch {hovered.result!.batch}
          </span>
          <span>{formatUsd(hovered.price!)}/GPU-hour</span>
        </div>
      )}
    </figure>
  );
}

type Scale = { scale: (value: number) => number; ticks: number[] };

function formatHoursTick(hours: number): string {
  if (hours < 1) return `${Math.round(hours * 60)} min`;
  if (hours < 48) return `${Number(hours.toPrecision(3))} h`;
  return `${Number((hours / 24).toPrecision(3))} d`;
}

/* Tick mantissas from coarse to fine; the first set that puts at least three
   ticks inside the domain is used, so narrow ranges still get a readable axis. */
const TICK_STEPS = [
  [1, 2, 5],
  [1, 1.5, 2, 3, 5, 7],
  [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8],
];

/** A log10 scale padded around the data, with round ticks inside it. */
function logScale(values: number[], from: number, to: number): Scale {
  let low = Math.min(...values);
  let high = Math.max(...values);
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
    for (let exponent = Math.floor(Math.log10(low)); exponent <= Math.ceil(Math.log10(high)); exponent++) {
      for (const step of steps) {
        const tick = Number((step * 10 ** exponent).toPrecision(3));
        if (tick >= low && tick <= high) ticks.push(tick);
      }
    }
    if (ticks.length >= 3) break;
  }
  return { scale, ticks };
}

type Label = {
  row: Row;
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
function placeLabels(rows: Row[], x: Scale, y: Scale, width: number): Label[] {
  const narrow = width < 560;
  const charWidth = 6.4;
  const plot: Box = {
    left: MARGIN.left,
    right: width - 4,
    top: 0,
    bottom: CHART_HEIGHT - MARGIN.bottom,
  };
  const dots = rows.map((row) => ({ x: x.scale(row.hours!), y: y.scale(row.cost!) }));
  const placed: Box[] = [];
  const order = [...rows].sort((a, b) => Number(b.pareto) - Number(a.pareto));

  return order.map((row) => {
    const text = narrow ? shortName(row.gpu.name) : row.gpu.name;
    const textWidth = text.length * charWidth;
    const px = x.scale(row.hours!);
    const py = y.scale(row.cost!);
    const candidates: Omit<Label, "row" | "text">[] = [
      { x: px + 9, y: py + 4, anchor: "start" },
      { x: px - 9, y: py + 4, anchor: "end" },
      { x: px, y: py - 11, anchor: "middle" },
      { x: px, y: py + 20, anchor: "middle" },
    ];
    const boxOf = (candidate: Omit<Label, "row" | "text">): Box => {
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
    return { row, text, ...choice };
  });
}

/* ---------- CSV export ---------- */

function downloadCsv(
  rows: Row[],
  model: string,
  context: number,
  tokenCount: number,
  providerLabel: string
) {
  const header = [
    "gpu",
    "model",
    "context",
    "tokens",
    "price_source",
    "status",
    "batch",
    "tokens_per_second",
    "mfu_pct",
    "board_clock_mfu_pct",
    "peak_allocated_gb",
    "usd_per_gpu_hour",
    "hours",
    "cost_usd",
    "seeds",
    "pareto",
  ];
  const lines = rows.map((row) =>
    [
      row.gpu.id,
      modelLabel(model),
      context,
      tokenCount,
      row.overridden ? `${providerLabel} (edited)` : providerLabel,
      row.result ? "measured" : row.cell ? "oom" : "not_run",
      row.result?.batch ?? "",
      row.result?.tps ?? "",
      row.result?.mfu ?? "",
      row.result?.mfuBoard ?? "",
      row.result?.vramGb ?? "",
      row.price ?? "",
      row.hours?.toFixed(4) ?? "",
      row.cost?.toFixed(4) ?? "",
      row.result?.seeds.join(" ") ?? "",
      row.pareto,
    ]
      .map((value) => {
        const text = String(value);
        return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
      })
      .join(",")
  );
  const blob = new Blob([[header.join(","), ...lines].join("\n") + "\n"], {
    type: "text/csv;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `olm-benchmarks-${model}-${context}.csv`;
  anchor.click();
  URL.revokeObjectURL(url);
}
