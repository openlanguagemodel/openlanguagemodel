import fs from "fs";
import path from "path";

/**
 * The model zoo, generated from the code by scripts/gen_model_zoo.py.
 *
 * The site reads the same JSON as docs/models/index.md, so family lists,
 * preset counts and parameter counts never need a hand edit here: rerun the
 * generator after adding a model family.
 */
export const ZOO_FILE = path.join(process.cwd(), "..", "docs", "models", "zoo.json");

export type ZooPreset = {
  name: string;
  summary: string;
  line: number;
  exported: boolean;
  params: { total: number; active: number | null; mtp: number };
  tied_embeddings: boolean | null;
  tags: string[];
  dims: {
    layers: number | null;
    hidden: number | null;
    context: number | null;
    vocab: number | null;
  };
};

export type ZooFamily = {
  name: string;
  class: string;
  module: string;
  organisation: string;
  source: string;
  line: number;
  exported: boolean;
  guide: string | null; // repo-relative path, e.g. "docs/models/llama-3.md"
  structure: string | null;
  tags: string[];
  presets: ZooPreset[];
};

export type Zoo = {
  note: string;
  summary: { families: number; presets: number; organisations: number };
  families: ZooFamily[];
};

export function loadZoo(): Zoo {
  return JSON.parse(fs.readFileSync(ZOO_FILE, "utf-8")) as Zoo;
}

/** Families grouped by organisation, in the zoo's order. */
export function familiesByOrganisation(zoo: Zoo): [string, ZooFamily[]][] {
  const groups = new Map<string, ZooFamily[]>();
  for (const family of zoo.families) {
    const list = groups.get(family.organisation) ?? [];
    list.push(family);
    groups.set(family.organisation, list);
  }
  return Array.from(groups.entries());
}

/** Format a parameter count like the generated page: 124.4M, 7.50B, 1.03T. */
export function formatParams(n: number | null): string {
  if (n === null) return "—";
  for (const [unit, scale] of [
    ["T", 1e12],
    ["B", 1e9],
    ["M", 1e6],
    ["K", 1e3],
  ] as const) {
    if (n >= scale) {
      const value = n / scale;
      return value < 10 ? `${value.toFixed(2)}${unit}` : `${value.toFixed(1)}${unit}`;
    }
  }
  return String(n);
}

/** Site route of a family's architecture guide ("docs/models/x.md" -> "/docs/models/x/"). */
export function guideHref(family: ZooFamily): string | null {
  if (!family.guide) return null;
  return `/${family.guide.replace(/\.md$/, "")}/`;
}

/** GitHub URL of the line that defines a family or one of its presets. */
export function sourceHref(repo: string, family: ZooFamily, line = family.line): string {
  return `${repo}/blob/main/${family.source}#L${line}`;
}
