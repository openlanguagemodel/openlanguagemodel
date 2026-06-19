"use client";

import { useState } from "react";
import Link from "next/link";
import ScrambleText from "./ScrambleText";

export type NavItem = { title: string; path: string };
export type NavGroup = { label?: string; items: NavItem[] };
export type NavTrack = { id: string; label: string; groups: NavGroup[] };

interface Props {
  nav: NavTrack[];
  current: string;
  activeTrack: string;
  meta: [string, string, string];
}

export default function DocsSidebar({ nav, current, activeTrack, meta }: Props) {
  const [mobileOpen, setMobileOpen] = useState(false);
  // Each track is collapsible; the track you are in starts expanded.
  const [open, setOpen] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(nav.map((t) => [t.id, t.id === activeTrack]))
  );
  const active = nav.find((t) => t.id === activeTrack) ?? nav[0];
  const toggle = (id: string) => setOpen((o) => ({ ...o, [id]: !o[id] }));

  return (
    <aside className={`docs-aside ${mobileOpen ? "nav-open" : ""}`}>
      <div>
        {/* Brand reflects the track you are currently reading. */}
        <Link href="/" className="brand">
          <div className="status-dot" />
          <ScrambleText text={active.label} />
        </Link>
        <button
          className="nav-toggle"
          aria-label="Toggle navigation"
          onClick={() => setMobileOpen((o) => !o)}
        >
          MENU
        </button>
        <nav className="docs-nav">
          {nav.map((track) => (
            <div className="docs-nav-track" key={track.id}>
              <button
                className={`docs-nav-track-header${
                  track.id === activeTrack ? " current" : ""
                }`}
                aria-expanded={open[track.id]}
                onClick={() => toggle(track.id)}
              >
                <span>{track.label}</span>
                <span className="docs-nav-caret">
                  {open[track.id] ? "–" : "+"}
                </span>
              </button>
              {open[track.id] &&
                track.groups.map((group, gi) => (
                  <div className="docs-nav-group" key={group.label ?? gi}>
                    {group.label && (
                      <div className="docs-nav-grouplabel">{group.label}</div>
                    )}
                    <ul>
                      {group.items.map((item) => (
                        <li key={item.path}>
                          <Link
                            href={`/docs/${item.path}/`}
                            className={item.path === current ? "nav-active" : ""}
                          >
                            {item.title}
                          </Link>
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
            </div>
          ))}
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
