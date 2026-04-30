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
   python -m coverage run --source=src/fax -m pytest
   python -m coverage report --fail-under=95
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
* ``far3d-ingest`` converts FAR3D ``lincheck`` rows into the FAX kinetic
  validation schema. A short FAR3D/NVHPC GPU DIII-D runtime artifact is now
  archived with its run-reported normalization, and the same case has a
  50/100/200/400-step run-length ladder. The 200-step ``lincheck`` rows agree
  with the 400-step reference below ``3e-5`` maximum relative growth-rate
  difference. A first ``bet0_f`` scan shows a monotonic finite-beta growth
  branch with the beta-zero run retained as a control. This is still not a
  kinetic growth-rate parity claim because it is one DIII-D input, not a
  matched multi-geometry FAR3D/CAS3D-K benchmark.
* CAS3D and converged FAR3D parity remain pending until runnable input/output
  pairs or portable benchmark builds are available.

Do not publish the existing ``v0.0.1`` draft until the tag points at the final
release commit and the full local release gate above is green.
