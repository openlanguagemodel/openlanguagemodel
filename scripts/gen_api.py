#!/usr/bin/env python3
"""Generate lightweight Markdown and Sphinx API references from ``src/olm``."""

from __future__ import annotations

import inspect
import pkgutil
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


def public_name(name: str) -> bool:
    return not name.startswith("_")


def signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "()"


def clean_doc(obj: Any) -> str:
    doc = inspect.getdoc(obj) or ""
    return doc.strip()


def first_sentence(obj: Any) -> str:
    doc = clean_doc(obj)
    if not doc:
        return ""
    return doc.split("\n\n", 1)[0].replace("\n", " ")


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


def class_methods(cls: type) -> list[tuple[str, Any]]:
    methods: list[tuple[str, Any]] = []
    for name, obj in inspect.getmembers(cls, inspect.isfunction):
        if not public_name(name):
            continue
        if getattr(obj, "__qualname__", "").startswith(f"{cls.__name__}."):
            methods.append((name, obj))
    return methods


def module_has_api(module: ModuleType) -> bool:
    classes, functions = local_public_members(module)
    return bool(classes or functions)


def module_markdown(module: ModuleType) -> str:
    lines = [f"# `{module.__name__}`", ""]
    module_doc = clean_doc(module)
    if module_doc:
        lines.extend([module_doc, ""])

    classes, functions = local_public_members(module)

    if functions:
        lines.extend(["## Functions", ""])
        for name, func in functions:
            lines.extend([f"### `{name}{signature(func)}`", ""])
            doc = clean_doc(func)
            if doc:
                lines.extend([doc, ""])

    if classes:
        lines.extend(["## Classes", ""])
        for name, cls in classes:
            lines.extend([f"### `{name}{signature(cls)}`", ""])
            doc = clean_doc(cls)
            if doc:
                lines.extend([doc, ""])

            methods = class_methods(cls)
            if methods:
                lines.extend(["#### Methods", ""])
                for method_name, method in methods:
                    lines.extend([f"- `{method_name}{signature(method)}`"])
                    method_doc = first_sentence(method)
                    if method_doc:
                        lines.extend([f"  {method_doc}"])
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def group_label(module_name: str) -> str:
    parts = module_name.split(".")
    if len(parts) <= 1:
        return "Top Level"
    labels = {
        "core": "Core",
        "data": "Data",
        "logging": "Logging",
        "models": "Models",
        "nn": "Neural Network Components",
        "plugins": "Plugins",
        "train": "Training",
    }
    return labels.get(parts[1], parts[1].title())


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
    write_rst_index(modules)

    print(f"Generated API reference for {len(modules)} modules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
