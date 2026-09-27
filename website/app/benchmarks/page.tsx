import type { Metadata } from "next";
import Link from "next/link";
import Sidebar from "../components/Sidebar";
import ScrambleText from "../components/ScrambleText";
import BenchmarkExplorer from "../components/BenchmarkExplorer";
import { SITE } from "../../site.config";
import { SITE_META, SITE_NAV_LINKS } from "../lib/siteNav";
import { jsonLd, pageMetadata } from "../lib/seo";
import {
  DATA,
  cellFor,
  formatDate,
  formatGb,
  formatInteger,
  formatPct,
  formatUsd,
  gpuLabel,
  modelLabel,
  noResultReason,
  providerById,
  selectedBatch,
  usdPerBillionTokens,
} from "../lib/benchmarks";

const HEADLINE_MODELS = ["150m", "500m"];
const HEADLINE_CONTEXT = 2048;
const BENCHMARK_TREE = `${SITE.repo}/tree/main/benchmarks/throughput`;
const CATALOG_URL = `${SITE.repo}/blob/main/benchmarks/throughput/data/benchmark_catalog.csv`;
const GPU_NAMES = DATA.gpus.map((gpu) => gpu.name);

export const metadata: Metadata = pageMetadata({
  title: "GPU Benchmarks",
  description:
    "Measured OLM training throughput, MFU, peak memory and cost per billion tokens on " +
    `${GPU_NAMES.slice(0, -1).join(", ")} and ${GPU_NAMES[GPU_NAMES.length - 1]}, ` +
    "with an interactive cost and time explorer.",
  path: "/benchmarks/",
});

export default function BenchmarksPage() {
  const community = providerById("runpod-community");
  const secure = providerById("runpod-secure");
  const scaling = DATA.scaling;
  const widest = scaling.rows[scaling.rows.length - 1];
  const maxScalingTps = Math.max(...scaling.rows.map((row) => row.gpus * scaling.rows[0].tps));
  const workload = scaling.workload;
  // Every GPU count has the same number of runs; say so only if that holds.
  const runCounts = new Set(scaling.rows.map((row) => row.replicates));
  const runs = runCounts.size === 1 ? scaling.rows[0].replicates : null;

  const structuredData = {
    "@context": "https://schema.org",
    "@type": "Dataset",
    name: "OpenLanguageModel single-GPU training throughput catalog",
    description:
      "Tokens per second, model FLOP utilization and peak memory for full training steps of 150M to 8B parameter Llama-3-style OLM models on rented GPUs, with dated prices.",
    url: `${SITE.url}/benchmarks/`,
    creator: { "@type": "Organization", name: SITE.name, url: SITE.url },
    distribution: {
      "@type": "DataDownload",
      encodingFormat: "text/csv",
      contentUrl: CATALOG_URL,
    },
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: jsonLd(structuredData) }}
      />
      <Sidebar navLinks={SITE_NAV_LINKS} meta={SITE_META} />
      <main>
        <header className="doc-header bench-header">
          <span className="hero-label">OLM Benchmarks</span>
          <h1>GPU Throughput and Cost</h1>
          <p className="bench-lede">
            How fast OLM trains on one rented GPU, what a billion training tokens
            costs, and how training scales across GPUs. Every measurement on this
            page is generated from the data in the repository; costs combine those
            measurements with dated price quotes.
          </p>
          <div className="hero-model-note bench-stats">
            <div className="hero-metric">
              <strong>{DATA.gpus.length}</strong>
              <span>
                GPUs measured, from {DATA.gpus[0].name} to{" "}
                {DATA.gpus[DATA.gpus.length - 1].name}
              </span>
            </div>
            <div className="hero-metric">
              <strong>{formatInteger(DATA.source.catalogRows)}</strong>
              <span>
                catalog rows over {DATA.models.length} model sizes and{" "}
                {DATA.contexts.length} context lengths
              </span>
            </div>
            <div className="hero-metric">
              <strong>{formatPct(widest.efficiency * 100)}</strong>
              <span>weak-scaling efficiency on {widest.gpus} A100 GPUs</span>
            </div>
          </div>
        </header>

        <section id="explorer">
          <div className="section-grid bench-stack">
            <div className="section-title">
              <ScrambleText text="01 — Explorer" className="block" />
            </div>
            <div className="section-content">
              <article className="entry">
                <h3>Price Your Training Run</h3>
                <p>
                  Choose a model, a context length, a token budget and a price
                  source. Each GPU shows its fastest measured batch, the time and
                  compute cost for your budget, and whether it sits on the
                  cost–time frontier. Open a GPU&rsquo;s batch sweep to see every
                  batch tried, including the ones that ran out of memory.
                </p>
                <BenchmarkExplorer />
              </article>
            </div>
          </div>
        </section>

        <section id="headline">
          <div className="section-grid bench-stack">
            <div className="section-title">
              <ScrambleText text="02 — At a Glance" className="block" />
            </div>
            <div className="section-content">
              <article className="entry">
                <h3>150M and 500M at 2,048 Tokens</h3>
                <p>
                  Cost per billion tokens uses RunPod on-demand rates captured on{" "}
                  {formatDate(secure.capturedAt)}, where every GPU in the catalog
                  was measured.
                </p>
                {HEADLINE_MODELS.map((model) => (
                  <div className="bench-headline" key={model}>
                    <h4>{modelLabel(model)} parameters</h4>
                    <div
                      className="bench-table-wrap"
                      role="region"
                      aria-label={`${modelLabel(model)} results`}
                      tabIndex={0}
                    >
                      <table className="bench-table">
                        <thead>
                          <tr>
                            <th scope="col">GPU</th>
                            <th scope="col" className="num">Batch</th>
                            <th scope="col" className="num">Tokens/s</th>
                            <th scope="col" className="num">MFU spec / board</th>
                            <th scope="col" className="num">Peak VRAM</th>
                            <th scope="col" className="num">$ / 1B tokens Community</th>
                            <th scope="col" className="num">$ / 1B tokens Secure</th>
                          </tr>
                        </thead>
                        <tbody>
                          {DATA.gpus.map((gpu) => {
                            const result = selectedBatch(cellFor(gpu.id, model, HEADLINE_CONTEXT));
                            const cost = (usd: number | undefined) =>
                              result && usd !== undefined
                                ? formatUsd(usdPerBillionTokens(result.tps, usd))
                                : "no quote";
                            return (
                              <tr key={gpu.id}>
                                <th scope="row">{gpuLabel(gpu)}</th>
                                {result ? (
                                  <>
                                    <td className="num">{result.batch}</td>
                                    <td className="num">{formatInteger(result.tps)}</td>
                                    <td className="num">
                                      {formatPct(result.mfu)} / {formatPct(result.mfuBoard)}
                                    </td>
                                    <td className="num">{formatGb(result.vramGb)}</td>
                                    <td className="num">{cost(community.quotes[gpu.id]?.usd)}</td>
                                    <td className="num">{cost(secure.quotes[gpu.id]?.usd)}</td>
                                  </>
                                ) : (
                                  <td colSpan={6} className="bench-status">
                                    {noResultReason(gpu.id, model, HEADLINE_CONTEXT)}
                                  </td>
                                )}
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ))}
                <p className="bench-footnote">
                  Best batch: the fastest batch that fit for both seeds. Tokens/s:
                  median of the two seeds. MFU: achieved training FLOPs over the
                  published dense Tensor Core peak, at NVIDIA&rsquo;s reference clock
                  (spec) and at the board&rsquo;s reported maximum clock (board).
                  Peak VRAM: <code className="inline">torch.cuda.max_memory_allocated()</code>.
                  Cost: compute only, excluding storage, setup, idle time and taxes.
                </p>
              </article>
            </div>
          </div>
        </section>

        <section id="scaling">
          <div className="section-grid bench-stack">
            <div className="section-title">
              <ScrambleText text="03 — Multi-GPU" className="block" />
            </div>
            <div className="section-content">
              <article className="entry">
                <h3>{formatPct(widest.efficiency * 100)} Weak Scaling on {widest.gpus} A100s</h3>
                <p>
                  From the OLM paper (
                  <a
                    className="hover-link"
                    href={scaling.source.paper}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    arXiv:2607.16669
                  </a>
                  ): a {Math.round(Number(workload.parameters) / 1e6)}M-parameter
                  Llama-3-style model trained with DistributedDataParallel on one
                  NVLink node of {String(workload.gpu_model).replace("NVIDIA ", "")} GPUs,
                  one {Number(workload.sequence_length).toLocaleString("en-US")}-token
                  sequence per GPU per step, {String(workload.precision).toUpperCase()}{" "}
                  autocast, {Number(workload.measured_steps)} measured steps
                  {runs !== null && <>, {runs} runs per GPU count</>}.
                </p>
                <figure className="bench-bars">
                  <figcaption>
                    Tokens/s, mean of {runs !== null ? `${runs} runs` : "all runs"}. The
                    outline marks perfect linear scaling from one GPU.
                  </figcaption>
                  {scaling.rows.map((row) => {
                    const ideal = row.gpus * scaling.rows[0].tps;
                    return (
                      <div className="bench-bar-row" key={row.gpus}>
                        <span className="bench-bar-label">
                          {row.gpus} GPU{row.gpus > 1 ? "s" : ""}
                        </span>
                        <span className="bench-bar-track" aria-hidden="true">
                          <span
                            className="bench-bar-ideal"
                            style={{ width: `${(100 * ideal) / maxScalingTps}%` }}
                          />
                          <span
                            className="bench-bar-fill"
                            style={{ width: `${(100 * row.tps) / maxScalingTps}%` }}
                          />
                        </span>
                        <span className="bench-bar-value">
                          {formatInteger(row.tps)}
                          <span className="bench-sub">{formatPct(row.efficiency * 100)}</span>
                        </span>
                      </div>
                    );
                  })}
                </figure>
                <div
                  className="bench-table-wrap"
                  role="region"
                  aria-label="Multi-GPU scaling results"
                  tabIndex={0}
                >
                  <table className="bench-table">
                    <thead>
                      <tr>
                        <th scope="col">GPUs</th>
                        <th scope="col" className="num">Tokens/s (mean ± std)</th>
                        <th scope="col" className="num">Per GPU</th>
                        <th scope="col" className="num">Efficiency</th>
                        <th scope="col" className="num">Peak memory per GPU</th>
                      </tr>
                    </thead>
                    <tbody>
                      {scaling.rows.map((row) => (
                        <tr key={row.gpus}>
                          <th scope="row">{row.gpus} × A100</th>
                          <td className="num">
                            {formatInteger(row.tps)} ± {formatInteger(row.tpsStd)}
                          </td>
                          <td className="num">{formatInteger(row.tps / row.gpus)}</td>
                          <td className="num">{formatPct(row.efficiency * 100)}</td>
                          <td className="num">{row.peakGb.toFixed(2)} GB</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="bench-footnote">
                  Efficiency is tokens/s on N GPUs divided by N times the
                  single-GPU tokens/s. These runs use an SXM A100, batch 1 per GPU
                  and a non-fused optimizer, so they are not directly comparable
                  with the single-GPU catalog above.
                </p>
              </article>
            </div>
          </div>
        </section>

        <section id="method">
          <div className="section-grid bench-stack">
            <div className="section-title">
              <ScrambleText text="04 — Method" className="block" />
            </div>
            <div className="section-content">
              <article className="entry">
                <h3>What Is Measured</h3>
                <div className="training-stack bench-method">
                  <div>
                    <strong>The step</strong>
                    <span>
                      A complete training step of a dense, tied-embedding
                      Llama-3-style OLM model: forward pass under BF16 autocast
                      (FP16 with loss scaling on GPUs without BF16 Tensor Cores),
                      shifted-token cross-entropy, backward pass and a fused
                      AdamW update, with FP32 weights, gradients and optimizer
                      state. No activation checkpointing, no torch.compile,
                      synthetic tokens.
                    </span>
                  </div>
                  <div>
                    <strong>The sweep</strong>
                    <span>
                      Batch sizes double at 2,048 tokens until memory runs out;
                      other contexts test three batches around the equivalent
                      tokens per step. Each point warms up until step time is
                      stable, then times 20 steps. Everything runs twice, with
                      seeds {DATA.protocolSeeds.join(" and ")}.
                    </span>
                  </div>
                  <div>
                    <strong>One number per cell</strong>
                    <span>
                      The fastest batch that completed for both seeds, as the
                      median of the two. A batch that fit one seed at the edge of
                      memory is shown in the batch sweep but never chosen.
                    </span>
                  </div>
                  <div>
                    <strong>MFU</strong>
                    <span>
                      Training FLOPs per token are 6N plus 12 × layers × width ×
                      context, divided by the published dense Tensor Core peak.
                      Sparse &ldquo;up to&rdquo; figures are never used.
                    </span>
                  </div>
                  <div>
                    <strong>Cost</strong>
                    <span>
                      Tokens ÷ tokens/s × price per GPU-hour. Performance was
                      measured on RunPod; prices from other providers are
                      projections at their dated quotes. Storage, setup, idle time,
                      evaluation, checkpointing and taxes are excluded.
                    </span>
                  </div>
                </div>
                <div className="hero-actions inline-actions">
                  <Link href="/docs/benchmarks/" className="button-link">
                    Benchmarks docs →
                  </Link>
                  <a
                    href={`${BENCHMARK_TREE}#readme`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="button-link quiet"
                  >
                    Full methodology ↗
                  </a>
                  <a
                    href={CATALOG_URL}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="button-link quiet"
                  >
                    Catalog CSV ↗
                  </a>
                </div>
              </article>
            </div>
          </div>
        </section>

        <section id="contribute">
          <div className="section-grid bench-stack">
            <div className="section-title">
              <ScrambleText text="05 — Contribute" className="block" />
            </div>
            <div className="section-content">
              <article className="entry">
                <h3>Add a GPU, Starting With a Free T4</h3>
                <p>
                  The Tesla T4 in a free Google Colab session is the next GPU
                  for the catalog. It has no BF16 Tensor Cores, so the harness
                  runs the same step with <code className="inline">--amp-dtype fp16</code>{" "}
                  and dynamic loss scaling. One notebook runs the whole suite,
                  both seeds, the validators and the exports; a merge script
                  adds the rows and the site and docs rebuild from them.
                </p>
                <div className="code-block compact">
                  <pre>
                    <code>
                      {`AMP_DTYPE=fp16 benchmarks/throughput/run_publication_suite.sh \\
    t4 /content/drive/MyDrive/olm-bench/t4 "$(command -v python)" "$PWD"`}
                    </code>
                  </pre>
                </div>
                <div className="hero-actions inline-actions">
                  <Link href="/docs/benchmarks/#contribute-a-gpu" className="button-link primary">
                    Step-by-step guide →
                  </Link>
                  <a
                    href={BENCHMARK_TREE}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="button-link"
                  >
                    Harness source ↗
                  </a>
                </div>
              </article>
            </div>
          </div>
        </section>
      </main>
    </>
  );
}
