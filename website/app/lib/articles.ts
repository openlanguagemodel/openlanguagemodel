import path from "path";

export const ARTICLES_DIR = path.join(process.cwd(), "..", "articles");

export type Article = {
  slug: string;
  title: string;
  description: string;
  date: string;
  keywords: string[];
};

export const ARTICLES: Article[] = [
  {
    slug: "build-gpt-2-style-language-model-from-scratch-pytorch",
    title: "Build a GPT-2 Style Language Model from Scratch in PyTorch",
    description:
      "A practical path from tokens and transformer blocks to training a GPT-style language model with OpenLanguageModel.",
    date: "2026-06-19",
    keywords: ["GPT-2", "PyTorch", "language model from scratch", "OLM"],
  },
  {
    slug: "how-transformer-blocks-work",
    title: "How Transformer Blocks Work: Attention, MLPs, Norms, and Residuals",
    description:
      "Understand transformer blocks as readable PyTorch components: attention, feed-forward layers, normalization, and residual wiring.",
    date: "2026-06-19",
    keywords: ["transformer blocks", "attention", "RMSNorm", "PyTorch"],
  },
  {
    slug: "train-125m-language-model-fineweb-edu",
    title: "How to Train a 125M Language Model on FineWeb-Edu",
    description:
      "A linkable training report for a roughly 125M parameter OLM run on FineWeb-Edu, including model shape, cost framing, and code.",
    date: "2026-06-19",
    keywords: ["FineWeb-Edu", "125M language model", "LLM training", "H100"],
  },
  {
    slug: "custom-transformer-architectures-pytorch",
    title: "Custom Transformer Architectures in PyTorch Without Rewriting the Training Loop",
    description:
      "Use OLM's Block system to swap attention, norms, feed-forward layers, and wiring while keeping the training path stable.",
    date: "2026-06-19",
    keywords: ["custom transformer architecture", "PyTorch", "ablation", "OLM"],
  },
  {
    slug: "ddp-vs-fsdp-language-model-training",
    title: "DDP vs FSDP for Language Model Training",
    description:
      "A concise guide to scaling language model training with DDP and FSDP in OpenLanguageModel.",
    date: "2026-06-19",
    keywords: ["DDP", "FSDP", "distributed LLM training", "PyTorch"],
  },
];

export function articlePath(slug: string): string {
  return `/articles/${slug}/`;
}

export function articleFile(slug: string): string {
  return path.join(ARTICLES_DIR, `${slug}.md`);
}

export function articleBySlug(slug: string): Article | undefined {
  return ARTICLES.find((article) => article.slug === slug);
}
