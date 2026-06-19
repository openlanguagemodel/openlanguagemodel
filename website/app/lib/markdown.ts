/* Markdown -> HTML pipeline for the docs.
 *
 * Renders the portable Markdown in docs/ into styled HTML, with:
 *  - GFM (tables, strikethrough, task lists)
 *  - GitHub-style alerts ("> [!NOTE]") -> styled callouts
 *  - mermaid code fences -> <div class="mermaid"> for client rendering
 *  - syntax highlighting (highlight.js classes, themed in globals.css)
 *  - rewriting of internal ".md" links to site routes (with basePath)
 */
import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkRehype from "remark-rehype";
import rehypeRaw from "rehype-raw";
import rehypeSlug from "rehype-slug";
import rehypeHighlight from "rehype-highlight";
import rehypeStringify from "rehype-stringify";
import { visit } from "unist-util-visit";
import { BASE_PATH } from "../../site.config";

/* eslint-disable @typescript-eslint/no-explicit-any */

function nodeText(node: any): string {
  if (!node) return "";
  if (node.type === "text") return node.value as string;
  if (node.children) return node.children.map(nodeText).join("");
  return "";
}

/** Convert ```mermaid fences into <div class="mermaid"> for client rendering. */
function rehypeMermaid() {
  return (tree: any) => {
    visit(tree, "element", (node: any, index: number | undefined, parent: any) => {
      if (node.tagName !== "pre" || !parent || index === undefined) return;
      const code = (node.children || []).find((c: any) => c.tagName === "code");
      if (!code) return;
      const className: string[] = code.properties?.className || [];
      if (!className.includes("language-mermaid")) return;
      parent.children[index] = {
        type: "element",
        tagName: "div",
        properties: { className: ["mermaid"] },
        children: [{ type: "text", value: nodeText(code) }],
      };
    });
  };
}

const ALERT = /^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*/i;

/** Convert GitHub-style alert blockquotes into styled callout divs. */
function rehypeAlerts() {
  return (tree: any) => {
    visit(tree, "element", (node: any) => {
      if (node.tagName !== "blockquote") return;
      const firstP = (node.children || []).find((c: any) => c.tagName === "p");
      if (!firstP || !firstP.children || !firstP.children.length) return;
      const firstText = firstP.children[0];
      if (!firstText || firstText.type !== "text") return;
      const m = firstText.value.match(ALERT);
      if (!m) return;

      const type = m[1].toLowerCase();
      firstText.value = firstText.value.replace(ALERT, "");
      // Drop a now-empty leading text/break left by the marker.
      while (
        firstP.children.length &&
        ((firstP.children[0].type === "text" && firstP.children[0].value === "") ||
          firstP.children[0].tagName === "br")
      ) {
        firstP.children.shift();
      }

      node.tagName = "div";
      node.properties = node.properties || {};
      node.properties.className = ["markdown-alert", `markdown-alert-${type}`];
      const title = type.charAt(0).toUpperCase() + type.slice(1);
      node.children.unshift({
        type: "element",
        tagName: "p",
        properties: { className: ["markdown-alert-title"] },
        children: [{ type: "text", value: title }],
      });
    });
  };
}

/** Rewrite internal ".md" links to site routes (e.g. /docs/guides/architecture/). */
function rehypeRewriteLinks(currentDir: string) {
  return (tree: any) => {
    visit(tree, "element", (node: any) => {
      if (node.tagName !== "a") return;
      const href = node.properties?.href;
      if (typeof href !== "string") return;
      if (/^(https?:|mailto:|#)/.test(href)) return;

      const [p, hash] = href.split("#");
      if (!p || !p.endsWith(".md")) return;

      let target = path.posix
        .normalize(path.posix.join(currentDir || ".", p))
        .replace(/\.md$/, "");
      target = target.replace(/(^|\/)index$/, "$1").replace(/\/$/, "");
      target = target.replace(/^\.\//, "").replace(/^\//, "");

      let url = `${BASE_PATH}/docs/${target}/`.replace(/([^:])\/{2,}/g, "$1/");
      if (hash) url += `#${hash}`;
      node.properties.href = url;
    });
  };
}

/** Rewrite relative image sources to the copied assets under /docs-assets/. */
function rehypeRewriteImages(currentDir: string) {
  return (tree: any) => {
    visit(tree, "element", (node: any) => {
      if (node.tagName !== "img") return;
      const src = node.properties?.src;
      if (typeof src !== "string") return;
      if (/^(https?:|data:|\/)/.test(src)) return; // external or already absolute

      let target = path.posix
        .normalize(path.posix.join(currentDir || ".", src))
        .replace(/^\.\//, "")
        .replace(/^\//, "");
      node.properties.src = `${BASE_PATH}/docs-assets/${target}`.replace(
        /([^:])\/{2,}/g,
        "$1/"
      );
    });
  };
}

export type RenderedDoc = { title: string; html: string };

export async function renderDoc(file: string, dir: string): Promise<RenderedDoc> {
  const raw = fs.readFileSync(file, "utf-8");
  const { content } = matter(raw);

  // Extract the first H1 as the page title and remove it from the body.
  let title = "";
  const lines = content.split("\n");
  const idx = lines.findIndex((l) => /^#\s+/.test(l));
  if (idx !== -1) {
    title = lines[idx].replace(/^#\s+/, "").trim();
    lines.splice(idx, 1);
  }
  const body = lines.join("\n").replace(/^\n+/, "");

  const out = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeRaw)
    .use(rehypeMermaid)
    .use(rehypeAlerts)
    .use(rehypeSlug)
    .use(rehypeRewriteLinks, dir)
    .use(rehypeRewriteImages, dir)
    .use(rehypeHighlight, { detect: false, ignoreMissing: true })
    .use(rehypeStringify)
    .process(body);

  return { title, html: String(out) };
}
