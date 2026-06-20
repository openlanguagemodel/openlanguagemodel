import type { Metadata } from "next";
import { SITE } from "../../site.config";

type PageMetadata = {
  title: string;
  description: string;
  path?: string;
  type?: "website" | "article";
};

export function absoluteUrl(path = "/"): string {
  if (/^https?:\/\//.test(path)) return path;
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  return `${SITE.url}${cleanPath}`.replace(/([^:])\/{2,}/g, "$1/");
}

export function pageMetadata({
  title,
  description,
  path = "/",
  type = "website",
}: PageMetadata): Metadata {
  const url = absoluteUrl(path);
  const image = SITE.ogImage;

  return {
    title,
    description,
    keywords: SITE.keywords,
    alternates: { canonical: url },
    openGraph: {
      title,
      description,
      url,
      siteName: SITE.name,
      type,
      images: [
        {
          url: image,
          width: 1200,
          height: 630,
          alt: "OpenLanguageModel: PyTorch LLM library for learning and ablations",
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: [image],
    },
  };
}

export function jsonLd(data: unknown): string {
  return JSON.stringify(data).replace(/</g, "\\u003c");
}
