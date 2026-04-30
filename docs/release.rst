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

v0.0.1 Readiness Notes
----------------------

The local release candidate should include the current external-code parity
state before publishing:

* ``stellgap-executable`` completes the public CTOK run only when the local
  executable accepts the documented ``xstgap 99 2048`` arguments and produces
  ``alfven_spec``.
* ``ae3d-global-mode`` is public AE3D matrix/eigenvector parity, not native
  FAX-vs-AE3D operator parity.
* CAS3D/FAR3D parity remains pending until runnable input/output pairs or
  portable executable builds are available.

Do not publish the existing ``v0.0.1`` draft until the tag points at the final
release commit and the full local release gate above is green.
