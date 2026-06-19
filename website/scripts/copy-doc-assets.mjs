// Copies image assets out of ../docs into public/docs-assets, preserving each
// file's docs-relative path. This lets Markdown image links (e.g.
// ![](gpt2-architecture.png) inside docs/learn/) resolve both on GitHub (the
// file sits next to the .md) and on the static site (served from /docs-assets/).
// Runs automatically via the predev / prebuild npm hooks.
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const here = path.dirname(fileURLToPath(import.meta.url));
const DOCS = path.join(here, "..", "..", "docs");
const OUT = path.join(here, "..", "public", "docs-assets");
const IMAGE_EXT = new Set([".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"]);

function walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(full);
    } else if (IMAGE_EXT.has(path.extname(entry.name).toLowerCase())) {
      const dest = path.join(OUT, path.relative(DOCS, full));
      fs.mkdirSync(path.dirname(dest), { recursive: true });
      fs.copyFileSync(full, dest);
    }
  }
}

fs.mkdirSync(OUT, { recursive: true });
if (fs.existsSync(DOCS)) walk(DOCS);
console.log("copied doc image assets -> public/docs-assets");
