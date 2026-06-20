#!/usr/bin/env python3
"""Generate lightweight Markdown and Sphinx API references from ``src/olm``."""

from __future__ import annotations

import inspect
import pkgutil
import re
import shutil
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DOCS = ROOT / "docs"
MD_OUT = DOCS / "generated"
RST_OUT = DOCS / "source" / "generated"
REPO_URL = "https://github.com/openlanguagemodel/openlanguagemodel/blob/main"

GROUPS = {
    "core": ("Core", "Distributed helpers, registries, and low-level utilities."),
    "data": ("Data", "Datasets, tokenizers, and OLM data loading."),
    "logging": ("Logging", "Experiment logging integrations."),
    "models": ("Models", "Implemented transformer model families and presets."),
    "nn": ("Neural Network Components", "Composable PyTorch modules for language-model architectures."),
    "plugins": ("Plugins", "Plugin extension points."),
    "train": ("Training", "Trainers, callbacks, optimizers, schedules, and device selection."),
}

DOC_SECTIONS = {
    "args": "Parameters",
    "arguments": "Parameters",
    "parameters": "Parameters",
    "returns": "Returns",
    "yields": "Yields",
    "raises": "Raises",
    "attributes": "Attributes",
    "forward": "Forward",
    "forward/training contract": "Forward / Training Contract",
    "implementation note": "Implementation Note",
    "iteration": "Iteration",
    "structure": "Structure",
    "example": "Example",
    "examples": "Examples",
}

SECTION_RE = re.compile(r"^([A-Za-z][A-Za-z /-]+):{1,2}\s*$")
FIELD_RE = re.compile(
    r"^(?P<name>\*\*?\w+|[\w.][\w.]*)(?:\s*\((?P<type>[^)]+)\))?:\s*(?P<desc>.*)$"
)
PYTHON_BLOCK_PREFIXES = (
    "if ",
    "elif ",
    "else:",
    "for ",
    "while ",
    "try:",
    "except",
    "finally:",
    "with ",
    "def ",
    "class ",
    "match ",
    "case ",
)


def public_name(name: str) -> bool:
    return not name.startswith("_")


def signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "()"


def clean_doc(obj: Any) -> str:
    doc = getattr(obj, "__doc__", None) or ""
    return inspect.cleandoc(doc).strip()


def dedent_lines(lines: list[str]) -> list[str]:
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return ["" for _ in lines]
    indent = min(len(line) - len(line.lstrip()) for line in non_empty)
    return [line[indent:] if len(line) >= indent else line for line in lines]


def doctest_to_python(lines: list[str]) -> str:
    out: list[str] = []
    for line in dedent_lines(lines):
        stripped = line.rstrip()
        if stripped.startswith(">>> "):
            out.append(stripped[4:])
        elif stripped == ">>>":
            out.append("")
        elif stripped.startswith("... "):
            out.append(stripped[4:])
        elif stripped == "...":
            out.append("")
        elif (
            stripped.endswith(":")
            and not stripped.startswith("#")
            and not stripped.startswith(PYTHON_BLOCK_PREFIXES)
        ):
            out.append(f"# {stripped}")
        else:
            out.append(stripped)
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def format_field_section(lines: list[str]) -> list[str]:
    body = dedent_lines(lines)
    items: list[dict[str, str]] = []
    freeform: list[str] = []

    for line in body:
        if not line.strip():
            continue
        match = FIELD_RE.match(line.strip())
        if match:
            items.append(
                {
                    "name": match.group("name"),
                    "type": match.group("type") or "",
                    "desc": match.group("desc").strip(),
                }
            )
        elif items:
            items[-1]["desc"] = f"{items[-1]['desc']} {line.strip()}".strip()
        else:
            freeform.append(line.strip())

    if not items:
        return freeform

    out: list[str] = []
    for item in items:
        label = f"`{item['name']}`"
        if item["type"]:
            label += f" (`{item['type']}`)"
        suffix = f": {item['desc']}" if item["desc"] else ""
        out.append(f"- {label}{suffix}")
    if freeform:
        out.extend(["", *freeform])
    return out


def format_doc(doc: str) -> str:
    if not doc:
        return ""

    lines = doc.splitlines()
    out: list[str] = []
    i = 0

    while i < len(lines):
        match = SECTION_RE.match(lines[i].strip())
        key = match.group(1).lower() if match else ""
        if not match or key not in DOC_SECTIONS:
            out.append(lines[i])
            i += 1
            continue

        title = DOC_SECTIONS[key]
        i += 1
        body: list[str] = []
        while i < len(lines):
            next_match = SECTION_RE.match(lines[i].strip())
            next_key = next_match.group(1).lower() if next_match else ""
            if next_match and next_key in DOC_SECTIONS:
                break
            body.append(lines[i])
            i += 1

        while out and not out[-1].strip():
            out.pop()
        out.extend(["", f"**{title}**", ""])

        if key in {"example", "examples"}:
            code = doctest_to_python(body)
            if code:
                out.extend(["```python", code, "```"])
        else:
            out.extend(format_field_section(body))
        out.append("")

    return "\n".join(out).strip()


def first_sentence(obj: Any) -> str:
    doc = clean_doc(obj)
    if not doc:
        return ""
    return doc.split("\n\n", 1)[0].replace("\n", " ")


def source_location(obj: Any) -> tuple[Path, int] | None:
    try:
        file = inspect.getsourcefile(obj)
        if inspect.ismodule(obj):
            line = 1
        else:
            _, line = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        return None
    if not file:
        return None
    return Path(file).resolve(), line


def is_olm_source(obj: Any) -> bool:
    location = source_location(obj)
    if location is None:
        return False
    path, _ = location
    try:
        path.relative_to(SRC)
    except ValueError:
        return False
    return True


def source_link(obj: Any) -> str:
    location = source_location(obj)
    if location is None:
        return ""
    path, line = location
    try:
        path.relative_to(SRC)
    except ValueError:
        return ""
    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError:
        return ""
    return f"[`{rel}:{line}`]({REPO_URL}/{rel}#L{line})"


def bases(cls: type) -> str:
    names = []
    for base in cls.__bases__:
        if base is object:
            continue
        if base.__module__.startswith("olm"):
            names.append(f"`{base.__module__}.{base.__name__}`")
        else:
            names.append(f"`{base.__name__}`")
    return ", ".join(names)


def property_signature(prop: property) -> str:
    if prop.fget is None:
        return ""
    annotation = inspect.signature(prop.fget).return_annotation
    if annotation is inspect.Signature.empty:
        return ""
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__name__", str(annotation)).replace("typing.", "")


def local_public_members(
    module: ModuleType,
) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]]]:
    classes: list[tuple[str, Any]] = []
    functions: list[tuple[str, Any]] = []
    exported = set(getattr(module, "__all__", []))

    for name, obj in inspect.getmembers(module):
        if not public_name(name):
            continue
        if inspect.isclass(obj):
            if obj.__module__ == module.__name__ or name in exported:
                classes.append((name, obj))
        elif inspect.isfunction(obj):
            if obj.__module__ == module.__name__ or name in exported:
                functions.append((name, obj))

    return classes, functions


def class_methods(cls: type) -> list[tuple[str, Any, str]]:
    methods: list[tuple[str, Any, str]] = []
    for name, obj in inspect.getmembers(cls, inspect.isfunction):
        if not public_name(name):
            continue
        if not is_olm_source(obj):
            continue
        if getattr(obj, "__qualname__", "").startswith(f"{cls.__name__}."):
            methods.append((name, obj, ""))

    if not any(name == "forward" for name, _, _ in methods) and hasattr(cls, "forward"):
        forward = getattr(cls, "forward")
        owner = getattr(forward, "__qualname__", "").split(".", 1)[0]
        if owner and owner != cls.__name__ and is_olm_source(forward):
            methods.insert(0, ("forward", forward, f"inherited from `{owner}`"))

    return methods


def class_properties(cls: type) -> list[tuple[str, property]]:
    props: list[tuple[str, property]] = []
    for name, obj in inspect.getmembers(cls):
        if (
            public_name(name)
            and isinstance(obj, property)
            and obj.fget is not None
            and is_olm_source(obj.fget)
        ):
            props.append((name, obj))
    return props


def module_has_api(module: ModuleType) -> bool:
    classes, functions = local_public_members(module)
    return bool(classes or functions)


def module_markdown(module: ModuleType) -> str:
    lines = [f"# `{module.__name__}`", ""]
    link = source_link(module)
    if link:
        lines.extend([f"Source: {link}", ""])

    module_doc = format_doc(clean_doc(module))
    if module_doc:
        lines.extend([module_doc, ""])

    classes, functions = local_public_members(module)

    if functions:
        lines.extend(["## Functions", ""])
        for name, func in functions:
            lines.extend([f"### `{name}{signature(func)}`", ""])
            link = source_link(func)
            if link:
                lines.extend([f"Source: {link}", ""])
            doc = format_doc(clean_doc(func))
            if doc:
                lines.extend([doc, ""])

    if classes:
        lines.extend(["## Classes", ""])
        for name, cls in classes:
            lines.extend([f"### `{name}{signature(cls)}`", ""])
            base_names = bases(cls)
            if base_names:
                lines.extend([f"**Bases:** {base_names}", ""])
            link = source_link(cls)
            if link:
                lines.extend([f"Source: {link}", ""])
            doc = format_doc(clean_doc(cls))
            if doc:
                lines.extend([doc, ""])

            props = class_properties(cls)
            if props:
                lines.extend(["#### Properties", ""])
                for prop_name, prop in props:
                    return_type = property_signature(prop)
                    label = f"`{prop_name}`"
                    if return_type:
                        label += f" -> `{return_type}`"
                    lines.append(f"- {label}")
                    prop_doc = format_doc(clean_doc(prop.fget)) if prop.fget is not None else ""
                    if prop_doc:
                        lines.append(f"  {prop_doc}")
                lines.append("")

            methods = class_methods(cls)
            if methods:
                lines.extend(["#### Methods", ""])
                for method_name, method, note in methods:
                    heading = f"##### `{method_name}{signature(method)}`"
                    if note:
                        heading += f" ({note})"
                    lines.extend([heading, ""])
                    link = source_link(method)
                    if link:
                        lines.extend([f"Source: {link}", ""])
                    method_doc = format_doc(clean_doc(method))
                    if method_doc:
                        lines.extend([method_doc, ""])

    return "\n".join(lines).rstrip() + "\n"


def group_label(module_name: str) -> str:
    parts = module_name.split(".")
    if len(parts) <= 1:
        return "Top Level"
    return GROUPS.get(parts[1], (parts[1].title(), ""))[0]


def write_markdown_index(modules: list[ModuleType]) -> None:
    grouped: dict[str, list[ModuleType]] = defaultdict(list)
    for module in modules:
        grouped[group_label(module.__name__)].append(module)

    lines = [
        "# API Reference",
        "",
        "Generated from the public Python API in `src/olm`.",
        "Each module page includes signatures, docstrings, and source-defined methods such as `forward()` where available.",
        "",
    ]

    for group in sorted(grouped):
        lines.extend([f"## {group}", ""])
        lines.extend(["| Module | Public API |", "|---|---|"])
        for module in sorted(grouped[group], key=lambda item: item.__name__):
            classes, functions = local_public_members(module)
            names = [name for name, _ in classes + functions]
            summary = ", ".join(f"`{name}`" for name in names[:8])
            if len(names) > 8:
                summary += f", +{len(names) - 8} more"
            path = f"generated/{module.__name__}.md"
            lines.append(f"| [`{module.__name__}`]({path}) | {summary} |")
        lines.append("")

    (DOCS / "api.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_package_pages(modules: list[ModuleType]) -> None:
    api_dir = DOCS / "api"
    api_dir.mkdir(parents=True, exist_ok=True)

    by_package: dict[str, list[ModuleType]] = defaultdict(list)
    for module in modules:
        parts = module.__name__.split(".")
        if len(parts) > 1:
            by_package[parts[1]].append(module)

    for key, package_modules in sorted(by_package.items()):
        label, desc = GROUPS.get(key, (key.title(), ""))
        lines = [
            f"# {label} API",
            "",
            desc or f"Public API modules under `olm.{key}`.",
            "",
            "## Modules",
            "",
            "| Module | Public API |",
            "|---|---|",
        ]
        for module in sorted(package_modules, key=lambda item: item.__name__):
            classes, functions = local_public_members(module)
            names = [name for name, _ in classes + functions]
            summary = ", ".join(f"`{name}`" for name in names[:8])
            if len(names) > 8:
                summary += f", +{len(names) - 8} more"
            rel = "../generated/" + module.__name__ + ".md"
            lines.append(f"| [`{module.__name__}`]({rel}) | {summary} |")
        lines.append("")
        (api_dir / f"{key}.md").write_text(
            "\n".join(lines).rstrip() + "\n", encoding="utf-8"
        )


def write_rst_index(modules: list[ModuleType]) -> None:
    lines = [
        "API Reference",
        "=============",
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "",
    ]
    for module in sorted(modules, key=lambda item: item.__name__):
        lines.append(f"   generated/{module.__name__}")
    lines.append("")
    (DOCS / "source" / "api.rst").write_text("\n".join(lines), encoding="utf-8")


def write_rst_module(module: ModuleType) -> None:
    title = module.__name__
    lines = [
        title,
        "=" * len(title),
        "",
        f".. automodule:: {title}",
        "   :members:",
        "   :undoc-members:",
        "   :show-inheritance:",
        "",
    ]
    (RST_OUT / f"{title}.rst").write_text("\n".join(lines), encoding="utf-8")


def discover_modules() -> list[ModuleType]:
    sys.path.insert(0, str(SRC))
    root = import_module("olm")
    modules: list[ModuleType] = []
    errors: list[str] = []

    for info in pkgutil.walk_packages(root.__path__, prefix="olm."):
        if ".__pycache__" in info.name:
            continue
        try:
            module = import_module(info.name)
        except Exception as exc:  # pragma: no cover - surfaced in script output
            errors.append(f"{info.name}: {exc}")
            continue
        if module_has_api(module):
            modules.append(module)

    if errors:
        print("Skipped modules that could not be imported:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)

    return sorted(modules, key=lambda item: item.__name__)


def main() -> int:
    modules = discover_modules()

    shutil.rmtree(MD_OUT, ignore_errors=True)
    shutil.rmtree(RST_OUT, ignore_errors=True)
    MD_OUT.mkdir(parents=True, exist_ok=True)
    RST_OUT.mkdir(parents=True, exist_ok=True)

    for module in modules:
        (MD_OUT / f"{module.__name__}.md").write_text(
            module_markdown(module), encoding="utf-8"
        )
        write_rst_module(module)

    write_markdown_index(modules)
    write_package_pages(modules)
    write_rst_index(modules)

    print(f"Generated API reference for {len(modules)} modules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
