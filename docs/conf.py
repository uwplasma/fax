from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fax import __version__  # noqa: E402


project = "fax"
author = "UWPlasma contributors"
copyright = f"{date.today().year}, {author}"  # noqa: A001
version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosectionlabel_prefix_document = True

if os.environ.get("READTHEDOCS") == "True":
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable", None),
    }
else:
    intersphinx_mapping = {}

try:
    import furo  # noqa: F401

    html_theme = "furo"
except Exception:
    html_theme = "alabaster"

html_title = "fax"
html_static_path = ["_static"]
html_context = {
    "display_github": True,
    "github_user": "uwplasma",
    "github_repo": "fax",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
