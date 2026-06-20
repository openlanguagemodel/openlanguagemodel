import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Link from "next/link";
import Sidebar from "../../components/Sidebar";
import Mermaid from "../../components/Mermaid";
import {
  ARTICLES,
  articleBySlug,
  articleFile,
  articlePath,
} from "../../lib/articles";
import { renderDoc } from "../../lib/markdown";
import { jsonLd, pageMetadata } from "../../lib/seo";
import { SITE_META, SITE_NAV_LINKS } from "../../lib/siteNav";
import { SITE } from "../../../site.config";

type Params = { slug: string };

export function generateStaticParams(): Params[] {
  return ARTICLES.map((article) => ({ slug: article.slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<Params>;
}): Promise<Metadata> {
  const { slug } = await params;
  const article = articleBySlug(slug);
  if (!article) return {};
  return pageMetadata({
    title: article.title,
    description: article.description,
    path: articlePath(article.slug),
    type: "article",
  });
}

export default async function ArticlePage({
  params,
}: {
  params: Promise<Params>;
}) {
  const { slug } = await params;
  const article = articleBySlug(slug);
  if (!article) notFound();

  const { title, html } = await renderDoc(articleFile(article.slug), "");
  const url = `${SITE.url}${articlePath(article.slug)}`;
  const structuredData = {
    "@context": "https://schema.org",
    "@type": "TechArticle",
    headline: article.title,
    name: article.title,
    description: article.description,
    url,
    datePublished: article.date,
    dateModified: article.date,
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
    keywords: article.keywords.join(", "),
    mainEntityOfPage: url,
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: jsonLd(structuredData) }}
      />
      <Sidebar navLinks={SITE_NAV_LINKS} meta={SITE_META} />
      <main>
        <header className="doc-header">
          <span className="hero-label">OLM Article · {article.date}</span>
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
            <Link href="/articles/">← Articles</Link>
            <Link href="/docs/getting-started/">Getting Started</Link>
            <Link href="/docs/guides/architecture/">Block System</Link>
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
