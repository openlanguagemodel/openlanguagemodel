"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import ScrambleText from "./ScrambleText";

interface NavLink {
  href: string;
  label: string;
}

interface SidebarProps {
  navLinks: NavLink[];
  meta: [string, string, string];
  brand?: string;
}

export default function Sidebar({ navLinks, meta, brand = "OLM" }: SidebarProps) {
  const [activeHref, setActiveHref] = useState(
    navLinks.find((l) => l.href.startsWith("#"))?.href ?? ""
  );
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const anchorLinks = navLinks.filter((l) => l.href.startsWith("#"));
    if (!anchorLinks.length) return;

    const sections = anchorLinks
      .map((l) => document.getElementById(l.href.slice(1)))
      .filter((el): el is HTMLElement => el !== null);

    const updateActive = () => {
      const trigger = window.scrollY + window.innerHeight * 0.3;
      let current = sections[0];
      for (const s of sections) {
        if (s.offsetTop <= trigger) current = s;
      }
      if (current) setActiveHref(`#${current.id}`);
    };

    window.addEventListener("scroll", updateActive, { passive: true });
    updateActive();
    return () => window.removeEventListener("scroll", updateActive);
  }, [navLinks]);

  return (
    <aside className={mobileOpen ? "nav-open" : ""}>
      <div>
        <Link href="/" className="brand">
          <div className="status-dot" />
          <ScrambleText text={brand} />
        </Link>
        <button
          className="nav-toggle"
          aria-label="Toggle navigation"
          onClick={() => setMobileOpen((o) => !o)}
        >
          MENU
        </button>
        <nav>
          <ul>
            {navLinks.map((link) => (
              <li key={link.href}>
                {link.href.startsWith("#") ? (
                  <a
                    href={link.href}
                    className={activeHref === link.href ? "nav-active" : ""}
                  >
                    {link.label}
                  </a>
                ) : (
                  <Link href={link.href}>{link.label}</Link>
                )}
              </li>
            ))}
          </ul>
        </nav>
      </div>
      <div className="meta-info">
        {meta.map((line, i) => (
          <p key={i}>{line}</p>
        ))}
      </div>
    </aside>
  );
}
