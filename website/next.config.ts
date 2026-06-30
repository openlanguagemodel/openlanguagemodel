import type { NextConfig } from "next";
import path from "path";
import { BASE_PATH } from "./site.config";

const nextConfig: NextConfig = {
  // Static HTML export -> deployable to GitHub Pages.
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  basePath: BASE_PATH || undefined,
  // The docs Markdown lives in ../docs (the repo root); include it in the
  // file-tracing root so the build can read it.
  outputFileTracingRoot: path.join(process.cwd(), ".."),
};

export default nextConfig;
