"""Sphinx configuration for the cloud-boot documentation site.

Build with `uv run --locked --group docs sphinx-build -W --keep-going -b html docs docs/_build/html`.
"""

from cloud_boot import __version__

project = "cloud-boot"
author = "cloud-boot contributors"
copyright = "cloud-boot contributors"
release = __version__
version = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build"]

# Docstrings write `name` for code; render it as code, and link it when it names a Python object.
# Unresolvable names such as parameters stay plain code, since `nitpicky` is off.
default_role = "py:obj"

# cloud-boot is a CLI, so the reference serves contributors: show private members too, in source
# order, which follows the call sequence.
autodoc_default_options = {
    "members": True,
    "private-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
# Add types only to documented arguments and returns, not a "Return type: None" on every method.
autodoc_typehints_description_target = "documented"

napoleon_google_docstring = True
napoleon_numpy_docstring = False

html_theme = "furo"
html_title = "cloud-boot"
