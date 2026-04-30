fax
===

``fax`` is a differentiable JAX research code for Alfvén continua, Alfvén
eigenmode screening, fast-ion diagnostics, continuum gaps, and toroidal/helical
coupling in stellarators.

The v0.0.1 implementation includes profile and metric continuum operators,
real ``boozmn`` local 3-D Boozer operators, reduced global radial eigenmodes,
differentiable isolated-mode and cluster objectives, reduced gyro-Landau
fast-ion closures, STELLGAP CTOK local-continuum/gap parity, and AE3D public
matrix/eigenmode parity. It is designed to consume ``vmec_jax`` equilibria and
``booz_xform_jax`` Boozer transforms as those end-to-end design loops mature.

Start with :doc:`quickstart` for commands, :doc:`model_inventory` for current
claim levels, :doc:`equations` for the implemented models, and
:doc:`publication_results` for the current figure/validation packet.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   configuration
   architecture
   model_inventory
   equations
   numerical_algorithms
   differentiability
   optimization
   user_workflows
   demonstrations
   research_examples
   publication_results
   research_grade_roadmap
   benchmark_manifest
   figures
   validation
   validation_traceability
   literature
   function_map
   release
   development
   api
