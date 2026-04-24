Configuration
=============

``fax`` uses TOML input files.

.. literalinclude:: ../examples/minimal.toml
   :language: toml

Top-level sections:

``[run]``
   Case name, selected physics model, and output directory.

``[equilibrium]``
   Equilibrium source and optional Boozer-coordinate files.

``[grid]``
   Radial and Fourier resolution.

``[physics]``
   Sound-coupling and fast-ion model flags.

``[numerics]``
   Precision, JIT/autodiff flags, and eigensolver choice.

