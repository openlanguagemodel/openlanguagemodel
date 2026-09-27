import type { Metadata } from "next";
import Link from "next/link";
import Sidebar from "../components/Sidebar";
import { SITE } from "../../site.config";
import { SITE_META, SITE_NAV_LINKS } from "../lib/siteNav";
import { jsonLd, pageMetadata } from "../lib/seo";
import {
  familiesByOrganisation,
  formatParams,
  guideHref,
  loadZoo,
  sourceHref,
  type ZooFamily,
} from "../lib/models";

// Everything on this page comes from docs/models/zoo.json, which
// scripts/gen_model_zoo.py generates from the code in src/olm/models.
const zoo = loadZoo();
const { families, presets, organisations } = zoo.summary;
const description = `${families} readable PyTorch model families and ${presets} named presets in OpenLanguageModel, from GPT-2 to trillion-parameter mixtures of experts, with parameter counts measured from the code.`;

export const metadata: Metadata = pageMetadata({
  title: "Model Families",
  description,
  path: "/models/",
});

function presetCount(family: ZooFamily): string {
  return `${family.presets.length} preset${family.presets.length === 1 ? "" : "s"}`;
}

function FamilyCard({ family }: { family: ZooFamily }) {
  const guide = guideHref(family);
  return (
    <div className="model-family">
      <span className="model-count">{presetCount(family)}</span>
      <span className="model-family-name">{family.name}</span>
      <p className="model-tags">{family.tags.join(" · ")}</p>
      <ul className="model-presets">
        {family.presets.map((preset) => (
          <li key={preset.name}>
            <a
              href={sourceHref(SITE.repo, family, preset.line)}
              target="_blank"
              rel="noopener noreferrer"
            >
              {preset.name}
            </a>
            <span className="model-preset-size">
              {formatParams(preset.params.total)}
              {preset.params.active !== null &&
                ` · ${formatParams(preset.params.active)} active`}
            </span>
          </li>
        ))}
      </ul>
      <span className="model-card-links">
        <a
          className="model-card-link"
          href={sourceHref(SITE.repo, family)}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={`View ${family.name} source code on GitHub`}
        >
          Source ↗
        </a>
        {guide && (
          <Link className="model-card-link" href={guide}>
            Architecture guide →
          </Link>
        )}
      </span>
    </div>
  );
}

export default function ModelsPage() {
  // One entry per guide page; a guide can cover several families.
  const guides = new Map<string, string[]>();
  for (const family of zoo.families) {
    const href = guideHref(family);
    if (href) guides.set(href, [...(guides.get(href) ?? []), family.name]);
  }

  const structuredData = {
    "@context": "https://schema.org",
    "@type": "CollectionPage",
    name: "OpenLanguageModel model families",
    description,
    url: `${SITE.url}/models/`,
    hasPart: zoo.families.map((family) => ({
      "@type": "SoftwareSourceCode",
      name: family.name,
      description: family.presets.map((p) => p.name).join(" · "),
      codeRepository: sourceHref(SITE.repo, family),
      programmingLanguage: "Python",
    })),
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: jsonLd(structuredData) }}
      />
      <Sidebar navLinks={SITE_NAV_LINKS} meta={SITE_META} />
      <main>
        <header className="doc-header">
          <span className="hero-label">OLM Models</span>
          <h1>Readable PyTorch Model Families</h1>
        </header>

        <article className="doc-body prose">
          <p>
            {families} model families from {organisations} organisations, with{" "}
            {presets}{" "}
            named presets that follow the published architecture of a
            released model. Each family is one source file that assembles the
            architecture from public OLM blocks, and every parameter count on
            this page comes from building the preset on PyTorch&apos;s{" "}
            <code>meta</code> device.
          </p>

          <div className="markdown-alert markdown-alert-important">
            <p className="markdown-alert-title">Important</p>
            <p>
              <strong>Presets are architectures, not checkpoints.</strong>{" "}
              A preset builds the released model&apos;s architecture with randomly
              initialised weights; OLM does not download or load pretrained
              weights. Use presets to train from scratch, to study or ablate an
              architecture, or to check shapes and sizes on the{" "}
              <code>meta</code> device.
            </p>
          </div>

          <h2>Architecture guides</h2>
          <p>
            These guides walk through a family&apos;s source section by section,
            map its arguments to the Hugging Face config, train a tiny version on
            a CPU, and list where OLM differs from the release.
          </p>
          <ul>
            {Array.from(guides.entries()).map(([href, names]) => (
              <li key={href}>
                <Link href={href}>{names.join(" and ")}</Link>
              </li>
            ))}
          </ul>

          <h2>All families</h2>
          <p>
            Sizes are total parameters; for mixtures of experts the second number
            is the parameters one token uses. The{" "}
            <Link href="/docs/models/">Model Zoo tables</Link> add layers, hidden
            size, context length, vocabulary and the layer mix of every preset.
          </p>
          {familiesByOrganisation(zoo).map(([organisation, group]) => (
            <div className="model-organisation" key={organisation}>
              <h3>{organisation}</h3>
              <div className="model-grid">
                {group.map((family) => (
                  <FamilyCard family={family} key={family.class} />
                ))}
              </div>
            </div>
          ))}
        </article>
      </main>
    </>
  );
}
