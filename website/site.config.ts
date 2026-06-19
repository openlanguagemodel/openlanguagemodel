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
  version: "2.2.0",
  tagline: "An open source LLM library for everyone.",
  description:
    "OpenLanguageModel (OLM) is a PyTorch-native LLM library for training language models, learning transformer architectures, and running custom architecture ablations.",
  url: "https://openlanguagemodel.github.io/openlanguagemodel",
  repo: "https://github.com/openlanguagemodel/openlanguagemodel",
  keywords: [
    "PyTorch LLM library",
    "train language models",
    "custom transformer architecture",
    "language model course",
    "single-node multi-GPU LLM training",
    "FineWeb-Edu training",
    "OpenLanguageModel",
    "OLM",
  ],
};
