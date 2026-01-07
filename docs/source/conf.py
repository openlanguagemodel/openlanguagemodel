import os
import sys

project = "olm"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_markdown_builder",
]

autosummary_generate = True
autosummary_ignore_module_all = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "imported-members": False,
}
autodoc_mock_imports = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "datasets",
    "tokenizers",
    "transformers",
]

templates_path = ["_templates"]
exclude_patterns = []

sys.path.insert(0, os.path.abspath("../../src"))

html_theme = "alabaster"
