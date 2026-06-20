import type { Metadata } from "next";
import Link from "next/link";
import DocsSidebar from "../components/DocsSidebar";
import { NAV } from "../lib/docs";
import { SITE } from "../../site.config";
import { jsonLd, pageMetadata } from "../lib/seo";

export const metadata: Metadata = pageMetadata({
  title: "Documentation",
  description:
    "OpenLanguageModel documentation for learning language models, training PyTorch LLMs, building custom transformer architectures, and using the OLM API.",
  path: "/docs/",
});

const paths = [
  {
    title: "New to language models",
    body: "Start with the from-scratch course, then hand off into the first train-and-generate tutorial.",
    href: "/docs/learn/",
    cta: "Start the course",
  },
  {
    title: "Ready to train",
    body: "Install OLM, load data, train a GPT-style model, save it, resume it, and generate from it.",
    href: "/docs/getting-started/",
    cta: "Get started",
  },
  {
    title: "Researching architectures",
    body: "Use the Block system, custom modules, raw PyTorch loops, and generated API reference for ablations.",
    href: "/docs/guides/architecture/",
    cta: "Read the Block guide",
  },
];

export default function DocsIndexPage() {
  const structuredData = {
    "@context": "https://schema.org",
    "@type": "CollectionPage",
    name: "OpenLanguageModel Documentation",
    description:
      "Documentation for learning, training, and researching transformer language models with OpenLanguageModel.",
    url: `${SITE.url}/docs/`,
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: jsonLd(structuredData) }}
      />
      <DocsSidebar
        nav={NAV}
        current="getting-started"
        activeTrack="learning"
        meta={["OLM Docs", "LICENSE: MIT", ""]}
      />

      <main>
        <header className="doc-header">
          <span className="hero-label">OLM Docs</span>
          <h1>Documentation</h1>
        </header>

        <article className="doc-body prose">
          <p>
            Pick the door that matches where you are. The same Markdown powers
            the website and the repository docs, so examples stay close to the
            code they describe.
          </p>

          <div className="path-grid">
            {paths.map((path) => (
              <Link className="path-card" href={path.href} key={path.href}>
                <span className="eyebrow">{path.cta}</span>
                <h2>{path.title}</h2>
                <p>{path.body}</p>
              </Link>
            ))}
          </div>
        </article>

        <footer>
          <div className="footer-brand">
            <div>OpenLanguageModel</div>
            <div className="footer-sub">OPEN SOURCE · MIT</div>
          </div>
          <div className="footer-links">
            <Link href="/">← Home</Link>
            <Link href="/docs/getting-started/">Getting Started</Link>
            <Link href="/docs/api/">API Reference</Link>
          </div>
          <div className="footer-meta">
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
