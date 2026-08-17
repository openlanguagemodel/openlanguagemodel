"""Print a small JSON usage snapshot for OLM.

This script uses public APIs where possible and GitHub's authenticated API when
available through `GH_TOKEN` or `GITHUB_TOKEN`. GitHub traffic fields require
repository access; if they are unavailable, the script returns `null` for them.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

REPO = "openlanguagemodel/openlanguagemodel"
PACKAGE = "openlanguagemodel"


def fetch_json(url: str, headers: dict[str, str] | None = None) -> Any:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def maybe_fetch_json(url: str, headers: dict[str, str] | None = None) -> Any | None:
    try:
        return fetch_json(url, headers)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def main() -> int:
    gh = github_headers()
    repo = fetch_json(f"https://api.github.com/repos/{REPO}", gh)
    pypi = fetch_json(f"https://pypi.org/pypi/{PACKAGE}/json")
    pypi_recent = maybe_fetch_json(
        f"https://pypistats.org/api/packages/{PACKAGE}/recent"
    )

    snapshot = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "github": {
            "stars": repo.get("stargazers_count"),
            "forks": repo.get("forks_count"),
            "watchers": repo.get("subscribers_count"),
            "open_issues": repo.get("open_issues_count"),
            "default_branch": repo.get("default_branch"),
            "traffic": {
                "clones": maybe_fetch_json(
                    f"https://api.github.com/repos/{REPO}/traffic/clones", gh
                ),
                "views": maybe_fetch_json(
                    f"https://api.github.com/repos/{REPO}/traffic/views", gh
                ),
                "referrers": maybe_fetch_json(
                    f"https://api.github.com/repos/{REPO}/traffic/popular/referrers",
                    gh,
                ),
                "paths": maybe_fetch_json(
                    f"https://api.github.com/repos/{REPO}/traffic/popular/paths",
                    gh,
                ),
            },
        },
        "pypi": {
            "version": pypi.get("info", {}).get("version"),
            "requires_python": pypi.get("info", {}).get("requires_python"),
            "recent_downloads": (
                pypi_recent.get("data") if isinstance(pypi_recent, dict) else None
            ),
        },
    }

    json.dump(snapshot, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
