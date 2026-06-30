# OpenLanguageModel — website

The OLM documentation website. It is a [Next.js](https://nextjs.org) app (static
export) that renders the **same Markdown** that lives in the repository's
[`../docs`](../docs) directory — so editing a doc updates both this site and the
version GitHub renders.

## Develop

```bash
cd website
npm install
npm run dev
```

Open <http://localhost:3000>. Pages auto-reload as you edit either the React code
here or the Markdown in `../docs`.

## How it works

- **Landing page** — `app/page.tsx` (the OLM overview).
- **Docs** — `app/docs/[...slug]/page.tsx` reads `../docs/**/*.md` at build time
  (`app/lib/markdown.ts`). The
  sidebar navigation is defined in `app/lib/docs.ts` (`NAV`). Docs render into
  the `.prose` style defined in `app/globals.css`.
- **API reference** — the pages under `../docs/api/` are generated from the
  library's docstrings by `../scripts/gen_api.py`. Re-run it after changing
  docstrings:

  ```bash
  python ../scripts/gen_api.py
  ```

- Markdown features supported: GFM tables, fenced code with syntax highlighting,
  GitHub-style alerts (`> [!NOTE]`), `<details>`, and Mermaid diagrams. Internal
  `*.md` links are rewritten to site routes automatically.

## Build (static export)

```bash
npm run build      # emits a static site to website/out/
```

Set `NEXT_PUBLIC_BASE_PATH` to control the deploy subpath:

- GitHub Pages **project** site (`…github.io/openlanguagemodel/`): `/openlanguagemodel` (the default in production).
- Custom domain or org-root site: set `NEXT_PUBLIC_BASE_PATH=""`.

## Deploy

Pushing to `main` triggers `.github/workflows/pages.yml`, which builds the export
and publishes it to GitHub Pages. **One-time setup:** in the repository's
*Settings → Pages*, set the source to **GitHub Actions**.
