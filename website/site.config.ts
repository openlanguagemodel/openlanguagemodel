/**
 * Site-wide configuration.
 *
 * BASE_PATH: GitHub Pages serves a project site from a subpath
 * (https://openlanguagemodel.github.io/openlanguagemodel/), so a production
 * build needs basePath="/openlanguagemodel" for links and assets to resolve.
 * Override with NEXT_PUBLIC_BASE_PATH (set it to "" if you deploy to a custom
 * domain or an org root site). Local `next dev` runs at the root.
 */
export const BASE_PATH =
  process.env.NEXT_PUBLIC_BASE_PATH ??
  (process.env.NODE_ENV === "production" ? "/openlanguagemodel" : "");

export const SITE = {
  name: "OpenLanguageModel",
  shortName: "OLM",
  tagline: "A PyTorch LLM library for learning, training, and ablations.",
  description:
    "OpenLanguageModel (OLM) is a modular PyTorch LLM library for building, training, teaching, and experimenting with transformer language models.",
  url: "https://openlanguagemodel.github.io/openlanguagemodel",
  repo: "https://github.com/openlanguagemodel/openlanguagemodel",
  ogImage: "https://openlanguagemodel.github.io/openlanguagemodel/og-image.svg",
  keywords: [
    "OpenLanguageModel",
    "OLM",
    "PyTorch LLM library",
    "train language models",
    "language model training",
    "custom transformer architecture",
    "transformer blocks",
    "FineWeb-Edu training",
    "distributed LLM training",
    "DDP",
    "FSDP",
    "language model course",
  ],
};
