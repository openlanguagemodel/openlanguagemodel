import type { Metadata } from "next";
import Link from "next/link";
import Sidebar from "../components/Sidebar";
import { ARTICLES, articlePath } from "../lib/articles";
import { SITE_META, SITE_NAV_LINKS } from "../lib/siteNav";
import { jsonLd, pageMetadata } from "../lib/seo";
import { SITE } from "../../site.config";

export const metadata: Metadata = pageMetadata({
  title: "Articles",
  description:
    "Technical articles about PyTorch LLM training, transformer architecture, FineWeb-Edu runs, DDP, FSDP, and OpenLanguageModel.",
  path: "/articles/",
});

export default function ArticlesPage() {
  const structuredData = {
    "@context": "https://schema.org",
    "@type": "ItemList",
    name: "OpenLanguageModel Articles",
    url: `${SITE.url}/articles/`,
    itemListElement: ARTICLES.map((article, index) => ({
      "@type": "ListItem",
      position: index + 1,
      url: `${SITE.url}${articlePath(article.slug)}`,
      name: article.title,
      description: article.description,
    })),
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
          <span className="hero-label">OLM Articles</span>
          <h1>PyTorch LLM Training Articles</h1>
        </header>

        <article className="doc-body prose">
          <p>
            Evergreen notes on building, training, teaching, and modifying
            transformer language models with OpenLanguageModel.
          </p>

          <div className="article-list">
            {ARTICLES.map((article) => (
              <Link
                className="article-card"
                href={articlePath(article.slug)}
                key={article.slug}
              >
                <span className="article-date">{article.date}</span>
                <h2>{article.title}</h2>
                <p>{article.description}</p>
              </Link>
            ))}
          </div>
        </article>
      </main>
    </>
  );
}
