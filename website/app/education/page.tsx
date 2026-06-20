import type { Metadata } from "next";
import Link from "next/link";
import Sidebar from "../components/Sidebar";
import { SITE } from "../../site.config";
import { SITE_META, SITE_NAV_LINKS } from "../lib/siteNav";
import { jsonLd, pageMetadata } from "../lib/seo";

const lessons = [
  "Set up a language-model lab",
  "Understand tokens and next-token prediction",
  "Build intuition for embeddings",
  "Learn attention step by step",
  "Assemble a transformer block",
  "Train and inspect a small language model",
];

export const metadata: Metadata = pageMetadata({
  title: "Teach Language Modelling With OLM",
  description:
    "Use OpenLanguageModel as a course or lab sequence for teaching tokens, embeddings, attention, transformer blocks, and language model training.",
  path: "/education/",
});

export default function EducationPage() {
  const structuredData = {
    "@context": "https://schema.org",
    "@type": "Course",
    name: "Learn Language Modelling From Scratch With OpenLanguageModel",
    description:
      "A course-style sequence for teaching language models by building tokens, embeddings, attention, transformer blocks, and training loops.",
    provider: {
      "@type": "Organization",
      name: "OpenLanguageModel",
      sameAs: SITE.repo,
    },
    url: `${SITE.url}/docs/learn/`,
    educationalLevel: "Undergraduate",
    teaches: lessons,
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
          <span className="hero-label">For Educators</span>
          <h1>Teach Language Modelling By Building One</h1>
        </header>

        <article className="doc-body prose">
          <p>
            OLM is designed for courses, labs, and reading groups that want
            students to inspect the actual pieces of a language model instead of
            only using pretrained checkpoints.
          </p>

          <div className="path-grid">
            <Link className="path-card" href="/docs/learn/">
              <span className="eyebrow">Course</span>
              <h2>Learn From Scratch</h2>
              <p>
                A walkable sequence from setup to tokens, embeddings, attention,
                transformer blocks, and training.
              </p>
            </Link>
            <Link className="path-card" href="/docs/tutorials/first-model/">
              <span className="eyebrow">Lab</span>
              <h2>Your First Language Model</h2>
              <p>
                A runnable train-and-generate exercise with save/load and
                sampling.
              </p>
            </Link>
            <Link className="path-card" href="/docs/guides/architecture/">
              <span className="eyebrow">Research</span>
              <h2>The Block System</h2>
              <p>
                A bridge from classroom transformer concepts to architecture
                ablations.
              </p>
            </Link>
          </div>

          <h2>Course Outline</h2>
          <ul>
            {lessons.map((lesson) => (
              <li key={lesson}>{lesson}</li>
            ))}
          </ul>

          <p>
            Start with the{" "}
            <Link href="/docs/learn/">Learn From Scratch course</Link>, then
            use the{" "}
            <Link href="/docs/tutorials/first-model/">first model tutorial</Link>{" "}
            and{" "}
            <Link href="/docs/guides/architecture/">Block System guide</Link>{" "}
            as reading material.
          </p>
        </article>
      </main>
    </>
  );
}
