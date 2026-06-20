import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Link from "next/link";
import DocsSidebar from "../../components/DocsSidebar";
import Mermaid from "../../components/Mermaid";
import { NAV, allDocPaths, resolveDoc, trackForPath } from "../../lib/docs";
import { renderDoc } from "../../lib/markdown";
import { SITE } from "../../../site.config";
import { jsonLd, pageMetadata } from "../../lib/seo";

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
  const docPath = slug.join("/");
  const { title, description } = await renderDoc(resolved.file, resolved.dir);
  return pageMetadata({
    title,
    description:
      description ||
      `OpenLanguageModel documentation for ${title}: PyTorch language model training, transformer architecture, and OLM APIs.`,
    path: `/docs/${docPath}/`,
    type: "article",
  });
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

  const { title, html, description } = await renderDoc(resolved.file, resolved.dir);
  const activeTrack = trackForPath(docPath);
  const trackLabel = NAV.find((t) => t.id === activeTrack)?.label ?? "OLM Docs";
  const url = `${SITE.url}/docs/${docPath}/`;
  const articleStructuredData = {
    "@context": "https://schema.org",
    "@type": "TechArticle",
    headline: title,
    name: title,
    description:
      description ||
      `OpenLanguageModel documentation for ${title}: PyTorch language model training, transformer architecture, and OLM APIs.`,
    url,
    author: {
      "@type": "Organization",
      name: "OpenLanguageModel",
      url: SITE.url,
    },
    publisher: {
      "@type": "Organization",
      name: "OpenLanguageModel",
      url: SITE.url,
    },
    mainEntityOfPage: url,
  };
  const structuredData =
    docPath === "learn"
      ? [
          articleStructuredData,
          {
            "@context": "https://schema.org",
            "@type": "Course",
            name: "Learn Language Modelling From Scratch With OpenLanguageModel",
            description:
              "A course-style sequence for learning tokens, embeddings, attention, transformer blocks, and language model training in PyTorch.",
            provider: {
              "@type": "Organization",
              name: "OpenLanguageModel",
              sameAs: SITE.repo,
            },
            url,
          },
        ]
      : articleStructuredData;

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: jsonLd(structuredData) }}
      />
      <DocsSidebar
        nav={NAV}
        current={docPath}
        activeTrack={activeTrack}
        meta={[trackLabel, "LICENSE: MIT", ""]}
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
