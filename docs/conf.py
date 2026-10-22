"""Sphinx configuration for FAX."""

from fax import __version__

project = "FAX"
author = "UWPlasma contributors"
version = release = __version__
extensions = ["sphinx.ext.autodoc", "sphinx.ext.mathjax", "sphinx.ext.napoleon"]
exclude_patterns = ["_build"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
html_theme = "furo"
html_title = "FAX"
