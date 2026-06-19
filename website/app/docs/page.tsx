import type { Metadata } from "next";
import Link from "next/link";
import DocsSidebar from "../components/DocsSidebar";
import Mermaid from "../components/Mermaid";
import { NAV } from "../lib/docs";
import { renderDoc } from "../lib/markdown";
import { SITE } from "../../site.config";

export const metadata: Metadata = {
  title: "Documentation",
  description: "OpenLanguageModel documentation for installation, training, architecture, models, and API reference.",
  alternates: { canonical: `${SITE.url}/docs/` },
  openGraph: {
    title: "OpenLanguageModel Documentation",
    description: "OpenLanguageModel documentation for installation, training, architecture, models, and API reference.",
    url: `${SITE.url}/docs/`,
    type: "article",
  },
};

export default async function DocsIndexPage() {
  const { title, html } = await renderDoc("../docs/index.md", "");
  const articleJsonLd = {
    "@context": "https://schema.org",
    "@type": "TechArticle",
    headline: title,
    url: `${SITE.url}/docs/`,
    isPartOf: {
      "@type": "WebSite",
      name: SITE.name,
      url: SITE.url,
    },
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(articleJsonLd) }}
      />
      <DocsSidebar
        nav={NAV}
        current="index"
        activeTrack="learning"
        meta={["OLM Docs", "LICENSE: MIT", `STATUS: v${SITE.version}`]}
      />

      <main>
        <header className="doc-header">
          <span className="hero-label">OLM Docs</span>
          <h1>{title}</h1>
        </header>

        <article
          className="doc-body prose"
          dangerouslySetInnerHTML={{ __html: html }}
        />

        <Mermaid />

        <footer>
          <div className="footer-brand">
            <div>OpenLanguageModel</div>
            <div className="footer-sub">OPEN SOURCE · MIT</div>
          </div>
          <div className="footer-links">
            <Link href="/">Home</Link>
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
