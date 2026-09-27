"""
docs/reference/defaults.md must match the code.

Every ``### `dotted.path` `` section of the page holds a table whose rows are
``| `argument` | `default` | notes |``. This test imports each path and checks
that the table lists every argument of its signature with the default the
code uses. A ``<!-- defaults-inherit: dotted.path -->`` line lets a section
take the rows it does not list from another section.
"""

import ast
import importlib
import inspect
import re
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "docs" / "reference" / "defaults.md"

SECTION_RE = re.compile(r"^### `([\w.]+)`\s*$")
INHERIT_RE = re.compile(r"^<!-- defaults-inherit: ([\w.]+) -->$")
ROW_RE = re.compile(r"^\| `(\*\*\w+|\w+)` \| ([^|]+) \|")


def _parse_page():
    sections = {}
    current = None
    for line in PAGE.read_text().splitlines():
        heading = SECTION_RE.match(line)
        if heading:
            current = {"rows": {}, "inherits": None}
            sections[heading.group(1)] = current
            continue
        if line.startswith("#"):
            current = None
        if current is None:
            continue
        inherit = INHERIT_RE.match(line)
        if inherit:
            current["inherits"] = inherit.group(1)
        row = ROW_RE.match(line)
        if row:
            current["rows"][row.group(1)] = row.group(2).strip()
    return sections


def _resolve(path):
    parts = path.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        for attribute in parts[split:]:
            obj = getattr(obj, attribute)
        return obj
    raise ImportError(path)


def _documented_value(cell):
    """Turn a table cell into the Python value it documents."""
    if cell == "required":
        return inspect.Parameter.empty
    literal = re.fullmatch(r"`([^`]+)`", cell)
    assert literal, f"default cell {cell!r} is not `required` or a `literal`"
    text = literal.group(1)
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text  # a class name such as `CrossEntropyLoss`


def _matches(documented, actual):
    if isinstance(documented, str) and inspect.isclass(actual):
        return documented == actual.__name__
    # 0 and 0.0 document the same default; True and 1 do not.
    numbers = all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in (documented, actual)
    )
    same_kind = numbers or type(documented) is type(actual)
    return same_kind and documented == actual


SECTIONS = _parse_page()


def test_page_documents_the_training_stack():
    for path in (
        "olm.train.Trainer",
        "olm.train.DDPTrainer",
        "olm.train.FSDPTrainer",
        "olm.train.AutoTrainer",
        "olm.train.CheckpointCallback",
        "olm.train.AdamW",
        "olm.train.WarmupCosineScheduler",
        "olm.train.CrossEntropyLoss",
        "olm.data.datasets.DataLoader",
    ):
        assert path in SECTIONS


@pytest.mark.parametrize("path", sorted(SECTIONS))
def test_documented_defaults_match_the_code(path):
    section = SECTIONS[path]
    signature = inspect.signature(_resolve(path))
    parameters = {
        name: parameter
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.kind is not inspect.Parameter.VAR_POSITIONAL
    }

    rows = dict(section["rows"])
    if section["inherits"]:
        for name, cell in SECTIONS[section["inherits"]]["rows"].items():
            if name in parameters:
                rows.setdefault(name, cell)

    expected_rows = {
        f"**{name}" if parameter.kind is inspect.Parameter.VAR_KEYWORD else name
        for name, parameter in parameters.items()
    }
    assert set(rows) == expected_rows, (
        f"{path}: undocumented {sorted(expected_rows - set(rows))}, "
        f"not in the signature {sorted(set(rows) - expected_rows)}"
    )

    for name, parameter in parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        documented = _documented_value(rows[name])
        if parameter.default is inspect.Parameter.empty:
            assert documented is inspect.Parameter.empty, f"{path}.{name} is required"
        else:
            assert _matches(documented, parameter.default), (
                f"{path}.{name}: the page says {rows[name]}, the code uses "
                f"{parameter.default!r}"
            )
