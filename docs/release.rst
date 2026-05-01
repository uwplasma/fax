Release Process
===============

FAX uses GitHub Releases and PyPI trusted publishing.

Version Metadata
----------------

For a release, update both:

* ``pyproject.toml`` project version
* ``src/fax/__init__.py`` ``__version__``

The v0.0.2 release uses ``0.0.2`` in both locations.

The PyPI distribution name is ``uwplasma-fax``. The import package and CLI are
still ``fax``.

FAX does not maintain a standalone ``CHANGELOG.md`` or ``MANIFEST.in``. Release
notes are attached to GitHub Releases, and package inclusion is controlled by
``pyproject.toml`` plus setuptools defaults.

Local Release Gate
------------------

Run this local gate before tagging any release, even when GitHub Actions are
available:

.. code-block:: bash

   python tools/generate_validation_figures.py
   python tools/qa_figures.py --figure-dir docs/_static
   python tools/qa_figures.py --figure-dir docs/_static/publication_suite
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

   gh release create v0.0.2 --title "FAX v0.0.2" --generate-notes

If Actions are unavailable, create a draft release and publish it once the
account-level Actions blocker has cleared.

v0.0.2 Readiness Notes
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
* ``examples/research/vmec_equilibria_catalog.py`` reads a local clone of
  ``landreman/vmec_equilibria`` and generates a broad VMEC-``wout``
  profile-continuum catalog over HSX, W7-X, NCSX, ARIES-CS, LHD, TJ-II, QHS,
  and ITER-like references. This is a geometry-matrix screening artifact, not
  full Boozer/global cross-code parity.
* ``examples/research/vmec_equilibria_boozer_global.py`` reads the public
  ``boozmn`` subset in that clone and runs HSX/Ku-Boozer cases through the
  global radial 3-D Boozer operator with residual, boundary-condition, and
  radial-stiffness variant metrics. This is still not AE3D/CAS3D/STELLGAP
  parity, but it closes the public-artifact Boozer/global adapter lane.
* ``examples/research/vmec_equilibria_live_boozer.py`` uses ``booz_xform_jax``
  to generate Boozer geometry from public W7-X/NCSX ``wout`` files and sends
  those geometries through the same FAX global operator, including
  coarse/reference truncation and runtime diagnostics.

Do not publish ``v0.0.2`` until the tag points at the final release commit and
the full local release gate above is green. The publish workflow runs from the
published GitHub release and then uploads the built source distribution and
wheel to PyPI through trusted publishing.
