import type { MetadataRoute } from "next";
import { SITE } from "../site.config";
import { ARTICLES, articlePath } from "./lib/articles";
import { allDocPaths } from "./lib/docs";

export const dynamic = "force-static";

function url(path: string): string {
  const clean = path.startsWith("/") ? path : `/${path}`;
  return `${SITE.url}${clean}`.replace(/([^:])\/{2,}/g, "$1/");
}

export default function sitemap(): MetadataRoute.Sitemap {
  const lastModified = new Date("2026-06-19");
  const staticRoutes = [
    { path: "/", priority: 1 },
    { path: "/docs/", priority: 0.9 },
    { path: "/articles/", priority: 0.85 },
    { path: "/models/", priority: 0.8 },
    { path: "/education/", priority: 0.8 },
  ];

  const docs = allDocPaths().map((path) => ({
    url: url(`/docs/${path}/`),
    lastModified,
    changeFrequency: "weekly" as const,
    priority: path === "learn" || path === "getting-started" ? 0.85 : 0.7,
  }));

  const articles = ARTICLES.map((article) => ({
    url: url(articlePath(article.slug)),
    lastModified: new Date(article.date),
    changeFrequency: "monthly" as const,
    priority: 0.75,
  }));

  return [
    ...staticRoutes.map((route) => ({
      url: url(route.path),
      lastModified,
      changeFrequency: "weekly" as const,
      priority: route.priority,
    })),
    ...docs,
    ...articles,
  ];
}
