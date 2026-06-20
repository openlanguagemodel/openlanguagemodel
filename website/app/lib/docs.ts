import fs from "fs";
import path from "path";

/** The docs Markdown lives in the repo's docs/ directory (one level up). */
export const DOCS_DIR = path.join(process.cwd(), "..", "docs");

export type NavItem = { title: string; path: string }; // path = docs-relative, no .md
export type NavGroup = { label?: string; items: NavItem[] };
export type NavTrack = { id: string; label: string; groups: NavGroup[] };

/**
 * The sidebar is split into three tracks for three audiences:
 *  - learning: teaches language modelling (two sub-sections, see below)
 *  - docs:     task-oriented guides for people who already know the concepts
 *  - api:      exhaustive, precise reference
 * `label` is also shown as the brand at the top of the sidebar while you are in
 * that track, so the reader always knows which area they are in.
 *
 * OLM Learning has two sub-sections for two very different beginners:
 *  - "Start Building": already knows LM basics, just wants to train a model.
 *  - "Learn From Scratch": a guided course from ~2nd-year-CS (Python + a little
 *    deep learning) to building language models — also runnable as a course.
 */
export const NAV: NavTrack[] = [
  {
    id: "learning",
    label: "OLM Learning",
    groups: [
      {
        label: "Start Building",
        items: [
          { title: "Getting Started", path: "getting-started" },
          { title: "Glossary & Concepts", path: "concepts" },
          { title: "Your First Language Model", path: "tutorials/first-model" },
          { title: "Modern Language Modelling", path: "tutorials/modern-language-modelling" },
          { title: "Custom Architectures", path: "tutorials/custom-architecture" },
          { title: "Colab Notebooks", path: "colab-notebooks" },
        ],
      },
      {
        label: "Learn From Scratch",
        items: [
          { title: "Course Overview", path: "learn" },
          { title: "0 · Set Up Your Lab", path: "learn/setup" },
          { title: "1 · What Is a Language Model?", path: "learn/what-is-a-language-model" },
          { title: "2 · Words as Vectors", path: "learn/words-as-vectors" },
          { title: "3 · Paying Attention", path: "learn/paying-attention" },
          { title: "4 · A Whole Transformer Block", path: "learn/a-transformer-block" },
          { title: "5 · How a Model Learns", path: "learn/how-a-model-learns" },
          { title: "Deep Dive · How Attention Works", path: "learn/attention-in-detail" },
        ],
      },
    ],
  },
  {
    id: "docs",
    label: "OLM Docs",
    groups: [
      {
        items: [
          { title: "The Block System", path: "guides/architecture" },
          { title: "Building Blocks", path: "guides/components" },
          { title: "Datasets & Training", path: "guides/datasets-and-training" },
          { title: "Distributed Training", path: "tutorials/distributed-training" },
          { title: "Experiment Tracking", path: "tutorials/experiment-tracking" },
          { title: "Roadmap", path: "roadmap" },
        ],
      },
    ],
  },
  {
    id: "api",
    label: "OLM API Reference",
    groups: [
      {
        items: [
          { title: "Overview", path: "api" },
          { title: "olm.nn", path: "api/nn" },
          { title: "olm.data", path: "api/data" },
          { title: "olm.train", path: "api/train" },
          { title: "olm.models", path: "api/models" },
          { title: "olm.core", path: "api/core" },
          { title: "olm.logging", path: "api/logging" },
        ],
      },
    ],
  },
];

/** Recursively collect every Markdown file under docs/. */
function walk(dir: string): string[] {
  const out: string[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walk(full));
    else if (entry.name.endsWith(".md")) out.push(full);
  }
  return out;
}

/**
 * Every doc route (docs-relative, no .md), derived from the filesystem so that
 * each Markdown file is statically generated even if it is not in the sidebar
 * (e.g. tutorials/index). The repo-root docs/index.md maps to the empty route
 * and is dropped — the site's landing page is app/page.tsx, not that file.
 */
export function allDocPaths(): string[] {
  return walk(DOCS_DIR)
    .map((file) =>
      path
        .relative(DOCS_DIR, file)
        .split(path.sep)
        .join("/")
        .replace(/\.md$/, "")
        .replace(/(^|\/)index$/, "$1")
        .replace(/\/$/, "")
    )
    .filter(Boolean);
}

/** Which track a given doc route belongs to (for the active-track brand). */
export function trackForPath(p: string): string {
  for (const t of NAV) {
    for (const g of t.groups) {
      if (g.items.some((i) => i.path === p)) return t.id;
    }
  }
  if (p.startsWith("api")) return "api";
  if (p.startsWith("guides")) return "docs";
  if (p === "learn" || p.startsWith("learn/")) return "learning";
  return "learning";
}

export function titleForPath(p: string): string {
  for (const t of NAV) {
    for (const g of t.groups) {
      for (const i of g.items) {
        if (i.path === p) return i.title;
      }
    }
  }
  return p;
}

/**
 * Resolve a docs-relative path (e.g. "guides/architecture" or "tutorials") to
 * the Markdown file and the directory it lives in (relative to docs/), which is
 * needed to resolve relative links from within that file.
 */
export function resolveDoc(p: string): { file: string; dir: string } | null {
  const flat = path.join(DOCS_DIR, p + ".md");
  if (fs.existsSync(flat)) {
    const dir = path.posix.dirname(p);
    return { file: flat, dir: dir === "." ? "" : dir };
  }
  const index = path.join(DOCS_DIR, p, "index.md");
  if (fs.existsSync(index)) {
    return { file: index, dir: p };
  }
  return null;
}
