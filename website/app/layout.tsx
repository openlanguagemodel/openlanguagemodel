import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Playfair_Display } from "next/font/google";
import "./globals.css";
import { SITE } from "../site.config";
import MascotLayer from "./components/MascotLayer";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  weight: ["300", "400"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
  weight: ["300", "400", "500", "700"],
});

const playfairDisplay = Playfair_Display({
  variable: "--font-playfair",
  subsets: ["latin"],
  weight: ["400", "600", "700"],
  style: ["normal", "italic"],
});

export const metadata: Metadata = {
  metadataBase: new URL(`${SITE.url}/`),
  applicationName: SITE.name,
  title: {
    default: "OpenLanguageModel: PyTorch LLM Library for Training Language Models",
    template: "%s — OpenLanguageModel",
  },
  description: SITE.description,
  keywords: SITE.keywords,
  authors: [
    { name: "Tavish Mankash" },
    { name: "Vardhaman Kalloli" },
    { name: "Keshava Prasad" },
  ],
  creator: "OpenLanguageModel",
  alternates: {
    canonical: SITE.url,
  },
  openGraph: {
    title: "OpenLanguageModel: PyTorch LLM Library for Training Language Models",
    description: SITE.description,
    url: SITE.url,
    siteName: SITE.name,
    type: "website",
    images: [
      {
        url: SITE.ogImage,
        width: 1200,
        height: 630,
        alt: "OpenLanguageModel: PyTorch LLM library for learning and ablations",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "OpenLanguageModel: PyTorch LLM Library",
    description: SITE.description,
    images: [SITE.ogImage],
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable} ${playfairDisplay.variable}`}
    >
      <body>
        <div className="site-wrapper">{children}</div>
        <MascotLayer />
      </body>
    </html>
  );
}
