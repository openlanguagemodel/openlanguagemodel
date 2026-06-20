import Link from "next/link";
import Sidebar from "./components/Sidebar";
import ScrambleText from "./components/ScrambleText";
import { BASE_PATH, SITE } from "../site.config";
import { SITE_META } from "./lib/siteNav";
import { jsonLd } from "./lib/seo";

const navLinks = [
  { href: "#overview", label: "Overview" },
  { href: "#start", label: "Start Here" },
  { href: "#quickstart", label: "Quickstart" },
  { href: "#training", label: "Training" },
  { href: "#architecture", label: "Architecture" },
  { href: "#models", label: "Models" },
  { href: "#roadmap", label: "Roadmap" },
  { href: "#contribute", label: "Contribute" },
  { href: "/docs/getting-started/", label: "Documentation →" },
];

const heroCode = `<span class="code-kw">from</span> olm.nn.structure <span class="code-kw">import</span> Block
<span class="code-kw">from</span> olm.nn.structure.combinators <span class="code-kw">import</span> Residual
<span class="code-kw">from</span> olm.nn.attention <span class="code-kw">import</span> GroupedQueryAttention
<span class="code-kw">from</span> olm.nn.feedforward <span class="code-kw">import</span> SwiGLUFFN
<span class="code-kw">from</span> olm.nn.norms <span class="code-kw">import</span> RMSNorm

llama3_block = <span class="code-fn">Block</span>([
    <span class="code-fn">Residual</span>(<span class="code-fn">Block</span>([
        <span class="code-fn">RMSNorm</span>(embed_dim, eps=1e-5),
        <span class="code-fn">GroupedQueryAttention</span>(
            embed_dim,
            num_heads,
            num_kv_heads,
            max_seq_len,
            dropout=dropout,
            rope_theta=rope_theta,
            use_bias=<span class="code-kw">False</span>,
        ),
    ])),
    <span class="code-fn">Residual</span>(<span class="code-fn">Block</span>([
        <span class="code-fn">RMSNorm</span>(embed_dim, eps=1e-5),
        <span class="code-fn">SwiGLUFFN</span>(
            embed_dim,
            hidden_dim=intermediate_size,
            dropout=dropout,
            bias=<span class="code-kw">False</span>,
        ),
    ])),
])`;

const llama3Code = `<span class="code-kw">from</span> olm.nn.structure <span class="code-kw">import</span> Block
<span class="code-kw">from</span> olm.nn.structure.combinators <span class="code-kw">import</span> Residual, Repeat
<span class="code-kw">from</span> olm.nn.attention <span class="code-kw">import</span> GroupedQueryAttention
<span class="code-kw">from</span> olm.nn.feedforward <span class="code-kw">import</span> SwiGLUFFN
<span class="code-kw">from</span> olm.nn.norms <span class="code-kw">import</span> RMSNorm
<span class="code-kw">from</span> olm.nn.embeddings <span class="code-kw">import</span> Embedding
<span class="code-kw">from</span> torch.nn <span class="code-kw">import</span> Linear

Llama3Style = <span class="code-fn">Block</span>([
    <span class="code-fn">Embedding</span>(vocab_size, embed_dim),
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
    <span class="code-fn">Linear</span>(embed_dim, vocab_size, bias=<span class="code-kw">False</span>)
])`;

const loopCode = `<span class="code-kw">for</span> inputs, targets <span class="code-kw">in</span> loader:
    logits = model(inputs)
    loss = loss_fn(logits, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()`;

const beginnerTrainingCode = `<span class="code-kw">import</span> torch

<span class="code-kw">from</span> olm.nn.blocks <span class="code-kw">import</span> LM
<span class="code-kw">from</span> olm.train <span class="code-kw">import</span> Trainer
<span class="code-kw">from</span> olm.data.tokenization <span class="code-kw">import</span> HFTokenizer
<span class="code-kw">from</span> olm.data.datasets <span class="code-kw">import</span> FineWebEduDataset, DataLoader

tok = <span class="code-fn">HFTokenizer</span>(<span class="code-str">"gpt2"</span>)
model = <span class="code-fn">LM</span>(
    tok.vocab_size,
    embed_dim=640,
    num_heads=10,
    num_layers=12,
    max_seq_len=1024,
    ff_multiplier=2.75,
)

dataset = <span class="code-fn">FineWebEduDataset</span>(tok, context_length=1024)
loader = <span class="code-fn">DataLoader</span>(dataset, batch_size=8, num_workers=4)
optimizer = torch.optim.<span class="code-fn">AdamW</span>(model.parameters(), lr=3e-4)
device = <span class="code-str">"cuda"</span> <span class="code-kw">if</span> torch.cuda.is_available() <span class="code-kw">else</span> <span class="code-str">"cpu"</span>

losses = <span class="code-fn">Trainer</span>(
    model,
    optimizer,
    loader,
    device,
    context_length=1024,
    use_amp=device == <span class="code-str">"cuda"</span>,
).train(epochs=1, max_steps=20_000)`;

const researcherCode = `<span class="code-kw">import</span> torch

<span class="code-kw">from</span> olm.nn.attention <span class="code-kw">import</span> AttentionBase

<span class="code-kw">class</span> <span class="code-fn">LocalWindowAttention</span>(AttentionBase):
    <span class="code-kw">def</span> <span class="code-fn">__init__</span>(self, embed_dim, num_heads, window=256):
        <span class="code-kw">super</span>().<span class="code-fn">__init__</span>(embed_dim, num_heads)
        self.window = window

    <span class="code-kw">def</span> <span class="code-fn">compute_attention</span>(self, q, k, v, mask=None):
        scores = (q @ k.transpose(-2, -1)) * self.scale
        seq = q.size(-2)
        pos = torch.<span class="code-fn">arange</span>(seq, device=q.device)
        local = (pos[:, None] - pos[None, :]).abs() &lt;= self.window
        causal = pos[:, None] &gt;= pos[None, :]
        scores = scores.masked_fill(~(local &amp; causal), float(<span class="code-str">"-inf"</span>))
        <span class="code-kw">if</span> mask <span class="code-kw">is not</span> None:
            scores = scores.masked_fill(mask == 0, float(<span class="code-str">"-inf"</span>))
        probs = self.dropout(scores.softmax(dim=-1))
        <span class="code-kw">return</span> probs @ v

attention = <span class="code-fn">LocalWindowAttention</span>(d_model, heads, window=256)
<span class="code-cmt"># Drop it into a Block, a custom model, or your PyTorch loop.</span>`;

const paths = [
  {
    title: "New to language models?",
    label: "Learn From Scratch",
    href: "/docs/learn/",
    desc: "Start with Colab, tokens, embeddings, attention, and training. Built for a second-year CS student who wants the whole thing to click.",
  },
  {
    title: "Know the basics?",
    label: "Train a model",
    href: "/docs/tutorials/first-model/",
    desc: "Build, train, save, reload, and sample from a small GPT-style model on Tiny Shakespeare in one runnable script.",
  },
  {
    title: "Doing ablations?",
    label: "Use the Block system",
    href: "/docs/guides/architecture/",
    desc: "Swap attention, norms, feed-forward layers, or wiring patterns without forking a monolithic model implementation.",
  },
];

const models = [
  {
    name: "GPT-2",
    count: "4 presets",
    sizes: "GPT2 · GPT2Medium · GPT2Large · GPT2XL",
    source: "src/olm/models/openai/gpt2.py",
  },
  {
    name: "Llama 2",
    count: "3 presets",
    sizes: "Llama2_7B · Llama2_13B · Llama2_70B",
    source: "src/olm/models/meta/llama2.py",
  },
  {
    name: "Llama 3.x",
    count: "5 presets",
    sizes: "Llama3_1_8B · Llama3_1_70B · Llama3_1_405B · Llama3_2_1B · Llama3_2_3B",
    source: "src/olm/models/meta/llama3.py",
  },
  {
    name: "Qwen 2.5",
    count: "7 presets",
    sizes: "Qwen2_5_0_5B · Qwen2_5_1_5B · Qwen2_5_3B · Qwen2_5_7B · Qwen2_5_14B · Qwen2_5_32B · Qwen2_5_72B",
    source: "src/olm/models/alibaba/qwen2.py",
  },
  {
    name: "Phi-3",
    count: "2 presets",
    sizes: "Phi3_5_Mini · Phi3_Small",
    source: "src/olm/models/microsoft/phi3.py",
  },
  {
    name: "Phi-4",
    count: "1 preset",
    sizes: "Phi4_14B",
    source: "src/olm/models/microsoft/phi4.py",
  },
  {
    name: "Gemma 2",
    count: "3 presets",
    sizes: "Gemma2_2B · Gemma2_9B · Gemma2_27B",
    source: "src/olm/models/google/gemma2.py",
  },
  {
    name: "OLMo",
    count: "1 preset",
    sizes: "OLMo_7B",
    source: "src/olm/models/allenai/olmo.py",
  },
  {
    name: "OPT",
    count: "1 preset",
    sizes: "OPT125M",
    source: "src/olm/models/facebook/opt.py",
  },
];

const roadmap = [
  { version: "v1.0", desc: "Foundation, core architectures, streaming data, single-GPU training.", done: true },
  { version: "v1.1", desc: "Flash/SDPA attention, RoPE/ALiBi variants, W&B logging, API docs.", done: true },
  { version: "v2.0", desc: "DDP, FSDP, and readable Mixture-of-Experts routing.", done: true },
  { version: "v2.1", desc: "AutoTrainer, hardware-aware training setup, and distributed/attention stability fixes.", done: true },
  { version: "v2.2", desc: "Website, SEO, mascot, API reference polish, and documentation refinement.", done: true },
  { version: "v3.0", desc: "Further training: SFT, LoRA, DPO, PPO/RLHF, GRPO-style RLVR, and evaluation recipes.", done: false },
  { version: "v4.0", desc: "Multi-node training and cluster support.", done: false },
];

export default function Home() {
  const structuredData = [
    {
      "@context": "https://schema.org",
      "@type": "WebSite",
      name: SITE.name,
      alternateName: "OLM",
      url: SITE.url,
      description: SITE.description,
      potentialAction: {
        "@type": "SearchAction",
        target: `${SITE.repo}/search?q={search_term_string}`,
        "query-input": "required name=search_term_string",
      },
    },
    {
      "@context": "https://schema.org",
      "@type": "SoftwareSourceCode",
      name: SITE.name,
      alternateName: "OLM",
      codeRepository: SITE.repo,
      url: SITE.url,
      programmingLanguage: "Python",
      runtimePlatform: "PyTorch",
      license: `${SITE.repo}/blob/main/LICENSE`,
      description:
        "A modular PyTorch LLM library for building, training, teaching, and experimenting with transformer language models.",
      keywords: SITE.keywords.join(", "),
    },
  ];

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: jsonLd(structuredData) }}
      />
      <Sidebar navLinks={navLinks} meta={SITE_META} />

      <main>
        <header id="overview" className="hero hero-home">
          <div className="hero-copy">
            <h1>
              <ScrambleText text={"OpenLanguage\nModel."} scrambleOnMount />
            </h1>
            <p>
              A modular PyTorch LLM library for building, training, teaching,
              and researching transformer language models. OLM does for LLMs
              what PyTorch did for deep learning: make the machinery readable,
              composable, and yours.
            </p>
            <div className="hero-actions" aria-label="Primary actions">
              <a
                href={SITE.repo}
                target="_blank"
                rel="noopener noreferrer"
                className="button-link primary"
              >
                ★ Star on GitHub
              </a>
              <Link href="/docs/getting-started/" className="button-link">
                Start building →
              </Link>
              <Link href="/docs/learn/" className="button-link quiet">
                New to LMs? Start here →
              </Link>
            </div>
            <div className="badge-row" aria-label="Project badges">
              <img
                alt="PyPI version"
                src="https://img.shields.io/pypi/v/openlanguagemodel?label=pypi"
              />
              <img
                alt="GitHub stars"
                src="https://img.shields.io/github/stars/openlanguagemodel/openlanguagemodel?style=social"
              />
              <img
                alt="Python 3.10+"
                src="https://img.shields.io/badge/python-3.10%2B-blue"
              />
              <img
                alt="MIT license"
                src="https://img.shields.io/badge/license-MIT-green"
              />
            </div>
            <div className="hero-model-note">
              <div className="hero-metric">
                <strong>30+</strong>
                <span>LM training runs from 100M to 1B scale.</span>
              </div>
              <div className="hero-metric">
                <strong>27</strong>
                <span>
                  Named model presets implemented, including GPT-2, Llama,
                  Qwen, Phi, Gemma, OLMo, and OPT.
                </span>
              </div>
            </div>
          </div>
          <div className="hero-code" aria-label="Llama 3 transformer block snippet">
            <div className="code-caption">llama3_block.py</div>
            <div
              className="code-block compact"
              dangerouslySetInnerHTML={{ __html: `<pre><code>${heroCode}</code></pre>` }}
            />
          </div>
        </header>

        <section id="start">
          <div className="section-grid">
            <div className="section-title">
              <ScrambleText text="01 — Start Here" className="block" />
            </div>
            <div className="section-content">
              <article className="entry">
                <h3>Choose Your Door</h3>
                <div className="path-grid">
                  {paths.map(({ title, label, href, desc }) => (
                    <Link href={href} key={href} className="path-card">
                      <span className="subtitle">{title}</span>
                      <strong>{label}</strong>
                      <p>{desc}</p>
                    </Link>
                  ))}
                </div>
              </article>
            </div>
          </div>
        </section>

        <section id="quickstart">
          <div className="section-grid">
            <div className="section-title">
              <ScrambleText text="02 — Quickstart" className="block" />
            </div>
            <div className="section-content">
              <article className="entry">
                <h3>Write the GPT-2 Architecture</h3>
                <p>
                  Start with a compact GPT-style model, then trace how embeddings,
                  transformer blocks, and the output head fit together. The same
                  pieces are available when you want to open the model and change
                  a part.
                </p>
                <img
                  src={`${BASE_PATH}/gpt2-diagram.png`}
                  alt="GPT-2 components mapped to OLM building blocks"
                  className="diagram-image"
                />
                <p>
                  <Link href="/docs/tutorials/first-model/" className="hover-link">
                    Train your first language model →
                  </Link>
                </p>
              </article>
            </div>
          </div>
        </section>

        <section id="training">
          <div className="section-grid">
            <div className="section-title">
              <ScrambleText text="03 — For Everyone" className="block" />
            </div>
            <div className="section-content">
              <article className="entry">
                <h3>OLM Is For Everyone</h3>
                <div className="training-grid">
                  <div className="training-panel">
                    <span className="subtitle">Beginner</span>
                    <h4>Train a Language Model With About $6</h4>
                    <p>
                      FineWeb-Edu streaming, GPT-2 tokenization, a roughly 125M
                      parameter model, and the training loop in one readable
                      script.
                    </p>
                    <div
                      className="code-block compact training-code"
                      dangerouslySetInnerHTML={{ __html: `<pre><code>${beginnerTrainingCode}</code></pre>` }}
                    />
                    <p className="training-footnote">
                      For the guided version with text generation and save/load,
                      see{" "}
                      <Link href="/docs/tutorials/first-model/" className="hover-link">
                        Your First Language Model →
                      </Link>
                    </p>
                  </div>
                  <div className="training-panel">
                    <span className="subtitle">Researcher</span>
                    <h4>Change Only What You Need To</h4>
                    <p>
                      Change the attention rule, test the idea, and leave the
                      rest of the training path alone.
                    </p>
                    <div
                      className="code-block compact training-code"
                      dangerouslySetInnerHTML={{ __html: `<pre><code>${researcherCode}</code></pre>` }}
                    />
                    <div className="training-stack">
                      <div>
                        <strong>Automatic Distributed Training Management</strong>
                        <span>OLM handles AMP, gradient accumulation, schedules, callbacks, DDP, FSDP, distributed sampling, rank-aware logging, metrics, and checkpointing when you want to scale.</span>
                      </div>
                      <div>
                        <strong>Fits PyTorch workflows</strong>
                        <span>use OLM modules inside existing PyTorch loops, scripts, notebooks, and research pipelines.</span>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="learning-callout">
                  <div>
                    <span className="subtitle">OLM is made for learning</span>
                    <h4>Teach Language Modelling By Building One</h4>
                  </div>
                  <p>
                    For courses, labs, and reading groups, OLM turns language
                    modelling into a sequence students can inspect: tokens,
                    embeddings, attention, blocks, and training.
                  </p>
                  <Link href="/docs/learn/" className="button-link primary">
                    Start the course →
                  </Link>
                </div>
              </article>
            </div>
          </div>
        </section>

        <section id="architecture">
          <div className="section-grid">
            <div className="section-title">
              <ScrambleText text="04 — Architecture" className="block" />
            </div>
            <div className="section-content">
              <article className="entry">
                <h3>A Real Llama-Style Model, Written as Blocks</h3>
                <p>
                  The core idea is separation: components say what happens;
                  <code className="inline">Block</code>,{" "}
                  <code className="inline">Residual</code>, and{" "}
                  <code className="inline">Repeat</code> say how those components
                  are wired. That makes architecture experiments local edits.
                </p>
                <div
                  className="code-block"
                  dangerouslySetInnerHTML={{ __html: `<pre><code>${llama3Code}</code></pre>` }}
                />
                <p>
                  <Link href="/docs/guides/architecture/" className="hover-link">
                    Learn the Block system →
                  </Link>
                </p>
              </article>
              <article className="entry">
                <h3>Bring Your Own Loop</h3>
                <p>
                  OLM components are ordinary PyTorch modules. You can train with
                  OLM's trainer, or call the model yourself inside the PyTorch
                  loop you already use.
                </p>
                <div
                  className="code-block compact"
                  dangerouslySetInnerHTML={{ __html: `<pre><code>${loopCode}</code></pre>` }}
                />
              </article>
            </div>
          </div>
        </section>

        <section id="models">
          <div className="section-grid">
            <div className="section-title">
              <ScrambleText text="05 — Models" className="block" />
            </div>
            <div className="section-content">
              <article className="entry">
                <h3>Reference Architectures, Reproduced With OLM</h3>
                <p>
                  The presets in <code className="inline">olm.models</code> are
                  readable implementations of familiar model families.
                </p>
                <div className="model-grid">
                  {models.map(({ name, count, sizes, source }) => (
                    <a
                      key={name}
                      className="model-family"
                      href={`${SITE.repo}/blob/main/${source}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      aria-label={`View ${name} source code on GitHub`}
                    >
                      <span className="subtitle">{name}</span>
                      <span className="model-count">{count}</span>
                      <p className="mono">{sizes}</p>
                      <span className="model-card-link">View source ↗</span>
                    </a>
                  ))}
                </div>
                <p>
                  <Link href="/docs/tutorials/modern-language-modelling/" className="hover-link">
                    Modern language modelling guide →
                  </Link>
                  {" "}
                  <Link href="/models/" className="hover-link">
                    Browse model source →
                  </Link>
                </p>
              </article>
            </div>
          </div>
        </section>

        <section id="roadmap">
          <div className="section-grid">
            <div className="section-title">
              <ScrambleText text="06 — Roadmap" className="block" />
            </div>
            <div className="section-content">
              <article className="entry">
                <h3>Roadmap</h3>
                <p>
                  OLM already supports complete language-model pretraining:
                  modern architectures, swappable components, streaming data,
                  checkpoints, mixed precision, and fast single-node multi-GPU
                  training.
                </p>
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

        <section id="contribute">
          <div className="section-grid">
            <div className="section-title">
              <ScrambleText text="07 — Contribute" className="block" />
            </div>
            <div className="section-content">
              <article className="entry">
                <h3>We're Open Source & Looking for Contributors</h3>
                <p>
                  Contributions are welcome across docs, examples, API reference,
                  model implementations, training stability, release polish, and
                  roadmap features. The docs and website render from the same
                  Markdown in this repository, so improvements travel everywhere.
                </p>
                <div className="hero-actions inline-actions">
                  <a
                    href={SITE.repo}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="button-link primary"
                  >
                    GitHub repository ↗
                  </a>
                  <Link href="/docs/api/" className="button-link">
                    API reference →
                  </Link>
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
            <Link href="/docs/learn/">Learn From Scratch</Link>
            <Link href="/docs/getting-started/">Start Building</Link>
            <Link href="/education/">Educators</Link>
            <Link href="/docs/guides/architecture/">Block System</Link>
            <Link href="/docs/api/">API Reference</Link>
          </div>
          <div className="footer-meta">
            <div>Status: v2.2</div>
            <a
              href={SITE.repo}
              target="_blank"
              rel="noopener noreferrer"
              className="hover-link"
            >
              Star on GitHub ↗
            </a>
          </div>
        </footer>
      </main>
    </>
  );
}
