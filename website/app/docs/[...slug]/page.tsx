import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Link from "next/link";
import DocsSidebar from "../../components/DocsSidebar";
import Mermaid from "../../components/Mermaid";
import { NAV, allDocPaths, resolveDoc, trackForPath } from "../../lib/docs";
import { renderDoc } from "../../lib/markdown";
import { SITE } from "../../../site.config";

type Params = { slug: string[] };

export function generateStaticParams(): Params[] {
  return allDocPaths().map((p) => ({ slug: p.split("/") }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<Params>;
}): Promise<Metadata> {
  const { slug } = await params;
  const resolved = resolveDoc(slug.join("/"));
  if (!resolved) return {};
  const { title } = await renderDoc(resolved.file, resolved.dir);
  const path = slug.join("/");
  const canonical = `${SITE.url}/docs/${path}/`;
  return {
    title,
    description: `${title} in the OpenLanguageModel documentation.`,
    alternates: { canonical },
    openGraph: {
      title,
      description: `${title} in the OpenLanguageModel documentation.`,
      url: canonical,
      type: "article",
    },
  };
}

export default async function DocPage({
  params,
}: {
  params: Promise<Params>;
}) {
  const { slug } = await params;
  const docPath = slug.join("/");
  const resolved = resolveDoc(docPath);
  if (!resolved) notFound();

  const { title, html } = await renderDoc(resolved.file, resolved.dir);
  const activeTrack = trackForPath(docPath);
  const trackLabel = NAV.find((t) => t.id === activeTrack)?.label ?? "OLM Docs";
  const articleJsonLd = {
    "@context": "https://schema.org",
    "@type": "TechArticle",
    headline: title,
    url: `${SITE.url}/docs/${docPath}/`,
    isPartOf: {
      "@type": "WebSite",
      name: SITE.name,
      url: SITE.url,
    },
    about: "OpenLanguageModel documentation",
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(articleJsonLd) }}
      />
      <DocsSidebar
        nav={NAV}
        current={docPath}
        activeTrack={activeTrack}
        meta={[trackLabel, "LICENSE: MIT", `STATUS: v${SITE.version}`]}
      />

      <main>
        <header className="doc-header">
          <span className="hero-label">{trackLabel}</span>
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
