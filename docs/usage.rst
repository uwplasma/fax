Usage
=====

Install with ``pip install uwplasma-fax`` (add ``[vmex]`` for the VMEC path).

Command line
------------

.. code-block:: bash

   fax run examples/ctok_continuum.toml --plot
   fax plot outputs/ctok_continuum

A case file has four tables; every key has a default:

.. code-block:: toml

   [geometry]
   source = "boozmn"          # boozmn | wout | vmec | circular  (vmec: optional ns, ftol, max_iter)
   path = "../tests/data/boozmn_ctok.nc"
   [modes]
   n = [2]
   m_max = 12
   [plasma]
   density = 1e19             # ions per m^3
   ion_mass = 2.0             # proton masses
   [run]
   task = "continuum"         # continuum | eigenmodes
   output = "../outputs/ctok_continuum"

Results go to ``results.npz`` and ``case.json`` in ``run.output``.

Python
------

.. code-block:: python

   import jax, fax

   geo = fax.read_boozmn("boozmn_ctok.nc")
   harmonics = fax.mode_family(2, 12)
   rho = fax.mass_density(1e19, ion_mass_amu=2.0)
   f_hz = fax.continuum(geo, harmonics, rho)          # (surfaces, harmonics)
   center, width = fax.gaps(f_hz)

   # derivative of the lowest global frequency with respect to density
   dfdrho = jax.grad(lambda r: fax.frequency_hz(geo, harmonics, r))(rho)
