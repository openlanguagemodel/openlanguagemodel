import Link from "next/link";
import Sidebar from "./components/Sidebar";
import ScrambleText from "./components/ScrambleText";
import { BASE_PATH, SITE } from "../site.config";

const navLinks = [
  { href: "#overview", label: "Overview" },
  { href: "#problem", label: "The Problem" },
  { href: "#quickstart", label: "Quickstart" },
  { href: "#simplicity", label: "Simplicity" },
  { href: "#models", label: "Models" },
  { href: "#extensibility", label: "Extensibility" },
  { href: "#roadmap", label: "Roadmap" },
  { href: "#contribute", label: "Contribute" },
  { href: "/docs/getting-started/", label: "Documentation →" },
];

const meta: [string, string, string] = [
  "OpenLanguageModel",
  "LICENSE: MIT",
  `STATUS: v${SITE.version}`,
];

const trainingCode = `<span class="code-kw">import</span> torch
<span class="code-kw">from</span> olm.nn.blocks <span class="code-kw">import</span> LM
<span class="code-kw">from</span> olm.train <span class="code-kw">import</span> Trainer
<span class="code-kw">from</span> olm.data.tokenization <span class="code-kw">import</span> HFTokenizer
<span class="code-kw">from</span> olm.data.datasets <span class="code-kw">import</span> LocalTextDataset, DataLoader

tokenizer = <span class="code-fn">HFTokenizer</span>(<span class="code-str">"gpt2"</span>)
device = <span class="code-str">"cuda"</span> <span class="code-kw">if</span> torch.cuda.is_available() <span class="code-kw">else</span> <span class="code-str">"cpu"</span>

<span class="code-cmt"># A complete GPT-style model in one line</span>
model = <span class="code-fn">LM</span>(tokenizer.vocab_size, 128, 4, 4, 128)
optimizer = torch.optim.<span class="code-fn">AdamW</span>(model.parameters(), 3e-4)

<span class="code-cmt"># Stream data &amp; train — AMP, scheduling, logging built in</span>
dataset = <span class="code-fn">LocalTextDataset</span>(<span class="code-str">"data/"</span>, tokenizer, 128)
loader = <span class="code-fn">DataLoader</span>(dataset, batch_size=8)
trainer = <span class="code-fn">Trainer</span>(model, optimizer, loader, device, 128, use_amp=<span class="code-kw">False</span>)

losses = trainer.<span class="code-fn">train</span>(epochs=1, max_steps=100)`;

const llama3Code = `<span class="code-kw">from</span> olm.nn.structure <span class="code-kw">import</span> Block
<span class="code-kw">from</span> olm.nn.structure.combinators <span class="code-kw">import</span> Residual, Repeat
<span class="code-kw">from</span> olm.nn.attention <span class="code-kw">import</span> GroupedQueryAttention
<span class="code-kw">from</span> olm.nn.feedforward <span class="code-kw">import</span> SwiGLUFFN
<span class="code-kw">from</span> olm.nn.norms <span class="code-kw">import</span> RMSNorm
<span class="code-kw">from</span> olm.nn.embeddings <span class="code-kw">import</span> Embedding
<span class="code-kw">from</span> olm.nn.blocks <span class="code-kw">import</span> OutputHead

embedding = <span class="code-fn">Embedding</span>(vocab_size, embed_dim)
Llama3Model = <span class="code-fn">Block</span>([
    embedding,
    <span class="code-fn">Repeat</span>(<span class="code-kw">lambda</span>: <span class="code-fn">Block</span>([
        <span class="code-fn">Residual</span>(<span class="code-fn">Block</span>([
            <span class="code-fn">RMSNorm</span>(embed_dim, eps=1e-5),
            <span class="code-fn">GroupedQueryAttention</span>(
                embed_dim, num_heads, num_kv_heads, max_seq_len, use_bias=<span class="code-kw">False</span>
            )
        ])),
        <span class="code-fn">Residual</span>(<span class="code-fn">Block</span>([
            <span class="code-fn">RMSNorm</span>(embed_dim, eps=1e-5),
            <span class="code-fn">SwiGLUFFN</span>(embed_dim, hidden_dim=intermediate_size, bias=<span class="code-kw">False</span>)
        ]))
    ]), num_layers),
    <span class="code-fn">RMSNorm</span>(embed_dim, eps=1e-5),
    <span class="code-fn">OutputHead</span>(embed_dim, vocab_size, tied_embedding=embedding, use_norm=<span class="code-kw">False</span>)
])`;

const activationCode = `<span class="code-kw">import</span> torch
<span class="code-kw">import</span> torch.nn.functional <span class="code-kw">as</span> F
<span class="code-kw">from</span> olm.nn.activations.base <span class="code-kw">import</span> ActivationBase

<span class="code-kw">class</span> <span class="code-fn">SwiGLU</span>(ActivationBase):
    <span class="code-kw">def</span> <span class="code-fn">forward</span>(self, x: torch.Tensor) -&gt; torch.Tensor:
        value, gate = x.chunk(2, dim=-1)
        <span class="code-kw">return</span> value * F.silu(gate)`;

const models = [
  { name: "LLAMA 3.x", sizes: "1B · 3B · 8B · 70B · 405B", href: "https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama3.py" },
  { name: "LLAMA 2", sizes: "7B · 13B · 70B", href: "https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/meta/llama2.py" },
  { name: "QWEN 2.5", sizes: "0.5B · 1.5B · 3B · 7B · 14B · 32B · 72B", href: "https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/alibaba/qwen2.py" },
  { name: "PHI-3 / PHI-4", sizes: "Mini · Small · 14B", href: "https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/microsoft" },
  { name: "GEMMA 2", sizes: "2B · 9B · 27B", href: "https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/google/gemma2.py" },
  { name: "GPT-2", sizes: "124M · 355M · 774M · 1.5B", href: "https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/openai/gpt2.py" },
  { name: "OLMo", sizes: "7B", href: "https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/allenai/olmo.py" },
  { name: "OPT", sizes: "125M", href: "https://github.com/openlanguagemodel/openlanguagemodel/blob/main/src/olm/models/facebook/opt.py" },
];

const roadmap = [
  { version: "v1.0", desc: "Foundation & core architectures.", done: true },
  { version: "v1.1", desc: "On-GPU optimization (Flash-Attention, torch.compile), ALiBi & RoPE scaling, W&B.", done: true },
  { version: "v2.0", desc: "Multi-GPU (DDP, FSDP) and Mixture-of-Experts.", done: true },
  { version: "v2.1", desc: "Bug fixes, AutoTrainer, model-family cleanup, and training stability.", done: true },
  { version: "v2.2", desc: "Stability, API reference, docs, website source, SEO foundations, and release readiness.", done: true },
  { version: "v3.0", desc: "Further training: SFT, LoRA, DPO, PPO, GRPO, and evaluation hooks.", done: false },
  { version: "v4.0", desc: "Multi-node training: cluster launch, fault tolerance, data sharding, pipeline and tensor parallelism.", done: false },
];

export default function Home() {
  return (
    <>
      <Sidebar navLinks={navLinks} meta={meta} />

      <main>
        {/* Hero */}
        <header id="overview" className="hero">
          <h1>
            <ScrambleText text={"OpenLanguage\nModel."} scrambleOnMount />
          </h1>
          <p>
            An open source LLM library for everyone. Does for LLMs what PyTorch
            did for deep learning.
          </p>
          <p style={{ marginTop: "1.5rem" }}>
            <Link href="/docs/getting-started/" className="hover-link" style={{ fontFamily: "var(--font-mono)", fontSize: "0.95rem" }}>
              Read the documentation →
            </Link>
          </p>
          <p style={{ marginTop: "1rem", fontFamily: "var(--font-mono)", fontSize: "0.85rem", color: "var(--text-muted)" }}>
            By Tavish Mankash, Vardhaman Kalloli, and Keshava Prasad
          </p>
        </header>

        {/* The Problem */}
        <section id="problem">
          <div className="section-grid">
            <div className="section-title">
              <ScrambleText text="01 — The Problem" className="block" />
            </div>
            <div className="section-content">
              <article className="entry">
                <h3>Why OLM Exists</h3>
                <p>
                  Typical LLM repositories run to thousands of lines of code. The
                  barrier to entry is enormous: domain specialization is required
                  just to get started, and there is no central, up-to-date
                  resource for training decent language models. Building language
                  models remains a niche, learned skill.
                </p>
                <p>
                  OLM is the answer:{" "}
                  <strong>simplified, modular, and transparent</strong>.
                </p>
              </article>
              <article className="entry">
                <h3>Two Audiences, One Library</h3>
                <div className="audience-grid">
                  <div>
                    <span className="subtitle">For Beginners</span>
                    <p>
                      Very easy to start. Train your own language models with
                      minimal setup. No domain specialization required.
                    </p>
                  </div>
                  <div>
                    <span className="subtitle">For Researchers</span>
                    <p>
                      The ability to go deep — precise architectural changes
                      without compromising ease of use, performance, or
                      customizability.
                    </p>
                  </div>
                </div>
              </article>
            </div>
          </div>
        </section>

        {/* Quickstart */}
        <section id="quickstart">
          <div className="section-grid">
            <div className="section-title">
              <ScrambleText text="02 — Quickstart" className="block" />
            </div>
            <div className="section-content">
              <article className="entry">
                <h3>Write the GPT-2 Architecture</h3>
                <img
                  src={`${BASE_PATH}/gpt2-diagram.png`}
                  alt="GPT-2 components mapped to OLM building blocks"
                  style={{ width: "100%", maxWidth: "720px", display: "block", marginTop: "1rem" }}
                />
                <p style={{ marginTop: "1.5rem" }}>
                  <Link href="/docs/getting-started/" className="hover-link">
                    Getting Started guide →
                  </Link>
                </p>
              </article>
            </div>
          </div>
        </section>

        {/* Simplicity */}
        <section id="simplicity">
          <div className="section-grid">
            <div className="section-title">
              <ScrambleText text="03 — Simplicity" className="block" />
            </div>
            <div className="section-content">
              <article className="entry">
                <h3>Training in a Few Lines</h3>
                <div
                  className="code-block"
                  dangerouslySetInnerHTML={{ __html: `<pre><code>${trainingCode}</code></pre>` }}
                />
                <p style={{ marginTop: "1.5rem" }}>
                  Models come from <code className="inline">olm.models</code>,
                  data pipelines from <code className="inline">olm.data</code>,
                  and training orchestration from{" "}
                  <code className="inline">olm.train</code>. Start with this
                  structure and gradually customize any part of it.
                </p>
                <p>
                  <Link href="/docs/getting-started/" className="hover-link">
                    Getting Started guide →
                  </Link>
                </p>
              </article>
            </div>
          </div>
        </section>

        {/* Replicated Models */}
        <section id="models">
          <div className="section-grid">
            <div className="section-title">
              <ScrambleText text="04 — Replicated Models" className="block" />
            </div>
            <div className="section-content">
              <article className="entry">
                <h3>Architectures, Reproduced from Blocks</h3>
                <p>
                  Reference architectures for the major model families — each
                  assembled from the same public building blocks you use:
                </p>
                <div className="model-grid">
                  {models.map(({ name, sizes, href }) => (
                    <a
                      key={name}
                      className="model-family"
                      href={href}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <span className="subtitle">{name}</span>
                      <p className="mono">{sizes}</p>
                    </a>
                  ))}
                </div>
              </article>
            </div>
          </div>
        </section>

        {/* Extensibility */}
        <section id="extensibility">
          <div className="section-grid">
            <div className="section-title">
              <ScrambleText text="05 — Extensibility" className="block" />
            </div>
            <div className="section-content">
              <article className="entry">
                <h3>Standardized Building Blocks</h3>
                <p>
                  All components — attention, feed-forward, norms — are modular.
                  Want a new loss function? Inherit from the base class and modify{" "}
                  <code className="inline">forward()</code>. No need to rewrite
                  the trainer or the data pipeline.
                </p>
                <p>
                  <Link href="/docs/architecture/" className="hover-link">
                    The Block System →
                  </Link>
                </p>
              </article>
              <article className="entry">
                <h3>Example: A Complete Llama 3 Architecture</h3>
                <div
                  className="code-block"
                  dangerouslySetInnerHTML={{ __html: `<pre><code>${llama3Code}</code></pre>` }}
                />
              </article>
              <article className="entry">
                <h3>Example: Custom Activation</h3>
                <div
                  className="code-block"
                  dangerouslySetInnerHTML={{ __html: `<pre><code>${activationCode}</code></pre>` }}
                />
              </article>
            </div>
          </div>
        </section>

        {/* Roadmap */}
        <section id="roadmap">
          <div className="section-grid">
            <div className="section-title">
              <ScrambleText text="06 — Status & Roadmap" className="block" />
            </div>
            <div className="section-content">
              <article className="entry">
                <h3>Current Status</h3>
                <p>
                  v2.2 is the stability and release-readiness pass: tied output
                  embeddings by default, cleaner model-family coverage, AutoTrainer,
                  improved API docs, tracked website source, and SEO foundations.
                </p>
              </article>
              <article className="entry">
                <h3>Roadmap</h3>
                <div className="roadmap-list">
                  {roadmap.map(({ version, desc, done }) => (
                    <div key={version} className={`roadmap-item${done ? " done" : ""}`}>
                      <span className="roadmap-version">{version}</span>
                      <span className="roadmap-desc">{desc}</span>
                    </div>
                  ))}
                </div>
              </article>
            </div>
          </div>
        </section>

        {/* Contribute */}
        <section id="contribute">
          <div className="section-grid">
            <div className="section-title">
              <ScrambleText text="07 — Contribute" className="block" />
            </div>
            <div className="section-content">
              <article className="entry">
                <h3>Call for Contributors</h3>
                <p>
                  We are looking for contributors for documentation & API
                  reference, feature additions, website & outreach, UX
                  enhancements, and major roadmap features.
                </p>
                <div style={{ marginTop: "2rem" }}>
                  <a
                    href={SITE.repo}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover-link"
                    style={{ fontSize: "1.2rem" }}
                  >
                    GITHUB REPOSITORY ↗
                  </a>
                </div>
              </article>
            </div>
          </div>
        </section>

        <footer>
          <div className="footer-brand">
            <div>OpenLanguageModel</div>
            <div className="footer-sub">OPEN SOURCE · MIT</div>
          </div>
          <div className="footer-links">
            <Link href="/docs/getting-started/">OLM Learning</Link>
            <Link href="/docs/architecture/">OLM Docs</Link>
            <Link href="/docs/api/">API Reference</Link>
          </div>
          <div className="footer-meta">
            <div>License: MIT</div>
            <div>Status: v{SITE.version}</div>
            <a
              href={SITE.repo}
              target="_blank"
              rel="noopener noreferrer"
              className="hover-link"
            >
              GitHub ↗
            </a>
          </div>
        </footer>
      </main>
    </>
  );
}
