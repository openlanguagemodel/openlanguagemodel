import type { MetadataRoute } from "next";
import { SITE } from "../site.config";
import { allDocPaths } from "./lib/docs";

export const dynamic = "force-static";

function page(path = "") {
  return `${SITE.url}${path}`;
}

export default function sitemap(): MetadataRoute.Sitemap {
  const now = new Date();
  const docs = allDocPaths().map((path) => ({
    url: page(`/docs/${path}/`),
    lastModified: now,
    changeFrequency: "weekly" as const,
    priority: path.startsWith("generated/") ? 0.45 : 0.75,
  }));

  return [
    {
      url: page("/"),
      lastModified: now,
      changeFrequency: "weekly",
      priority: 1,
    },
    {
      url: page("/docs/"),
      lastModified: now,
      changeFrequency: "weekly",
      priority: 0.9,
    },
    ...docs,
  ];
}
