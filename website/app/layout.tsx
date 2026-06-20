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
    default: "OpenLanguageModel (OLM) — PyTorch LLM Library",
    template: "%s — OpenLanguageModel",
  },
  description: SITE.description,
  keywords: SITE.keywords,
  authors: [
    { name: "Tavish Mankash" },
    { name: "Vardhaman Kalloli" },
    { name: "Keshava Prasad" },
  ],
  alternates: {
    canonical: SITE.url,
  },
  openGraph: {
    title: "OpenLanguageModel (OLM)",
    description: SITE.description,
    url: SITE.url,
    siteName: SITE.name,
    type: "website",
    images: [
      {
        url: "/og-image.svg",
        width: 1200,
        height: 630,
        alt: "OpenLanguageModel",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "OpenLanguageModel (OLM)",
    description: SITE.description,
    images: ["/og-image.svg"],
  },
  robots: {
    index: true,
    follow: true,
  },
};

const websiteJsonLd = {
  "@context": "https://schema.org",
  "@type": "WebSite",
  name: SITE.name,
  alternateName: SITE.shortName,
  url: SITE.url,
  description: SITE.description,
};

const softwareJsonLd = {
  "@context": "https://schema.org",
  "@type": "SoftwareSourceCode",
  name: SITE.name,
  alternateName: SITE.shortName,
  codeRepository: SITE.repo,
  programmingLanguage: "Python",
  runtimePlatform: "PyTorch",
  license: "https://github.com/openlanguagemodel/openlanguagemodel/blob/main/LICENSE",
  version: SITE.version,
  description: SITE.description,
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
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(websiteJsonLd) }}
        />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(softwareJsonLd) }}
        />
        <div className="site-wrapper">{children}</div>
        <MascotLayer />
      </body>
    </html>
  );
}
