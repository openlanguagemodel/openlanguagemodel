import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
PAGES = ["why-olm.md", "faq.md", "comparison.md"]
PYTHON_FENCE = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


def _snippets():
    cases = []
    for page in PAGES:
        text = (DOCS / page).read_text()
        for match in PYTHON_FENCE.finditer(text):
            line = text[: match.start()].count("\n") + 1
            cases.append(pytest.param(match.group(1), id=f"{page}:{line}"))
    return cases


@pytest.mark.parametrize("code", _snippets())
def test_orientation_page_snippet_runs(code):
    # Every Python block on the orientation pages must run as written on CPU.
    exec(compile(code, "<doc snippet>", "exec"), {"__name__": "__main__"})
