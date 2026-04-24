Release Process
===============

FAX uses GitHub Releases and PyPI trusted publishing.

Version Metadata
----------------

For a release, update both:

* ``pyproject.toml`` project version
* ``src/fax/__init__.py`` ``__version__``

The v0.0.1 release uses ``0.0.1`` in both locations.

The PyPI distribution name is ``uwplasma-fax``. The import package and CLI are
still ``fax``.

Local Release Gate
------------------

Until GitHub Actions resume, this local gate is mandatory:

.. code-block:: bash

   python tools/generate_validation_figures.py
   python tools/qa_figures.py
   python -m ruff check .
   python -m mypy src/fax
   python -m pytest -q
   sphinx-build -W -b html docs docs/_build/html
   python -m build
   python -m twine check dist/*

GitHub And PyPI
---------------

The publish workflow is ``.github/workflows/publish.yml``. It runs on published
GitHub releases and uses PyPI trusted publishing via
``pypa/gh-action-pypi-publish``.

When Actions are available:

.. code-block:: bash

   gh release create v0.0.1 --title "FAX v0.0.1" --notes-file .github/release_notes/v0.0.1.md

If Actions are unavailable, create a draft release and publish it once the
account-level Actions blocker has cleared.
