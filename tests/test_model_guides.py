import re
import textwrap
from pathlib import Path

import pytest

import olm.models

ROOT = Path(__file__).resolve().parents[1]
GUIDES = sorted((ROOT / "docs" / "models").glob("*.md"))

# A source excerpt is a link to a line range on GitHub, a blank line, then the
# code block. The block must be exactly those lines of the file (dedented), so a
# code change that moves or edits them fails here instead of leaving the guide
# quoting code that no longer exists.
EXCERPT = re.compile(
    r"^\[[^\]\n]*\]\(https://github\.com/openlanguagemodel/openlanguagemodel/"
    r"blob/main/(?P<path>[^#)\s]+)#L(?P<start>\d+)-L(?P<end>\d+)\)\n\n"
    r"```python\n(?P<code>.*?)```",
    re.M | re.S,
)
PYTHON_BLOCK = re.compile(r"^```python\n(?P<code>.*?)^```", re.M | re.S)
ZOO_MARKER = re.compile(r"<!--\s*model-zoo:\s*([\w\s]+?)\s*-->")


def _excerpts():
    for guide in GUIDES:
        for match in EXCERPT.finditer(guide.read_text(encoding="utf-8")):
            yield pytest.param(
                match, id=f"{guide.name}:{match['path']}#L{match['start']}"
            )


def _runnable_blocks():
    for guide in GUIDES:
        text = guide.read_text(encoding="utf-8")
        excerpt_spans = [m.span("code") for m in EXCERPT.finditer(text)]
        for index, match in enumerate(PYTHON_BLOCK.finditer(text)):
            if match.span("code") in excerpt_spans:
                continue
            yield pytest.param(guide, match["code"], id=f"{guide.name}:block{index}")


@pytest.mark.parametrize("match", list(_excerpts()))
def test_guide_source_excerpts_match_the_code(match):
    lines = (ROOT / match["path"]).read_text(encoding="utf-8").splitlines()
    start, end = int(match["start"]), int(match["end"])
    assert 1 <= start <= end <= len(lines)

    expected = textwrap.dedent("\n".join(lines[start - 1 : end])) + "\n"
    assert match["code"] == expected


@pytest.mark.parametrize("guide,code", list(_runnable_blocks()))
def test_guide_python_examples_run(guide, code):
    exec(compile(code, str(guide), "exec"), {"__name__": "__main__"})


@pytest.mark.parametrize(
    "guide", [g for g in GUIDES if g.name != "index.md"], ids=lambda g: g.name
)
def test_guides_register_existing_families_with_the_zoo(guide):
    names = [
        name
        for match in ZOO_MARKER.finditer(guide.read_text(encoding="utf-8"))
        for name in match.group(1).split()
    ]

    assert names
    for name in names:
        assert name in olm.models.__all__
