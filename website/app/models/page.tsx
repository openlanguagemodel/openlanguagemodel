import type { Metadata } from "next";
import Link from "next/link";
import Sidebar from "../components/Sidebar";
import { SITE } from "../../site.config";
import { SITE_META, SITE_NAV_LINKS } from "../lib/siteNav";
import { jsonLd, pageMetadata } from "../lib/seo";

const families = [
  ["GPT-2", "GPT2 · GPT2Medium · GPT2Large · GPT2XL", "src/olm/models/openai/gpt2.py"],
  ["Llama 2", "Llama2_7B · Llama2_13B · Llama2_70B", "src/olm/models/meta/llama2.py"],
  ["Llama 3.x", "Llama3_1_8B · Llama3_1_70B · Llama3_1_405B · Llama3_2_1B · Llama3_2_3B", "src/olm/models/meta/llama3.py"],
  ["Qwen 2.5", "0.5B · 1.5B · 3B · 7B · 14B · 32B · 72B", "src/olm/models/alibaba/qwen2.py"],
  ["Phi-3", "Phi3_5_Mini · Phi3_Small", "src/olm/models/microsoft/phi3.py"],
  ["Phi-4", "Phi4_14B", "src/olm/models/microsoft/phi4.py"],
  ["Gemma 2", "Gemma2_2B · Gemma2_9B · Gemma2_27B", "src/olm/models/google/gemma2.py"],
  ["OLMo", "OLMo_7B", "src/olm/models/allenai/olmo.py"],
  ["OPT", "OPT125M", "src/olm/models/facebook/opt.py"],
] as const;

export const metadata: Metadata = pageMetadata({
  title: "Model Families",
  description:
    "Source-linked model families implemented in OpenLanguageModel, including GPT-2, Llama, Qwen, Phi, Gemma, OLMo, and OPT.",
  path: "/models/",
});

export default function ModelsPage() {
  const structuredData = {
    "@context": "https://schema.org",
    "@type": "CollectionPage",
    name: "OpenLanguageModel model families",
    description:
      "Readable PyTorch implementations of GPT-2, Llama, Qwen, Phi, Gemma, OLMo, and OPT model families.",
    url: `${SITE.url}/models/`,
    hasPart: families.map(([name, sizes, source]) => ({
      "@type": "SoftwareSourceCode",
      name,
      description: sizes,
      codeRepository: `${SITE.repo}/blob/main/${source}`,
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
            OLM model presets are not hidden runtimes. Each family links to the
            source file that assembles the architecture from public blocks.
          </p>

          <div className="model-grid">
            {families.map(([name, sizes, source]) => (
              <a
                className="model-family"
                href={`${SITE.repo}/blob/main/${source}`}
                key={name}
                target="_blank"
                rel="noopener noreferrer"
              >
                <span className="subtitle">{name}</span>
                <p className="mono">{sizes}</p>
                <span className="model-card-link">View source ↗</span>
              </a>
            ))}
          </div>

          <p>
            For the concepts behind these implementations, read the{" "}
            <Link href="/docs/tutorials/modern-language-modelling/">
              Modern Language Modelling guide
            </Link>
            .
          </p>
        </article>
      </main>
    </>
  );
}
