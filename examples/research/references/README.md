The `qa_ae3d_potential.npz` physical-potential table is derived from the QA
`QA_35harmonics_scale0_001.npy` in the data archive by Knyazev, Paul, Lachmann,
Hyder, Goodman, Czekanski and Spong (2026), DOI
[10.5281/zenodo.21299908](https://doi.org/10.5281/zenodo.21299908).
The reference data are distributed under the archive's **CC BY 4.0** license.
The conversion preserves the signed potentials, radial samples and harmonic
labels; source-member and equilibrium SHA256 values are in JSON metadata.
No AE3D matrix, source code or particle trajectory is included.

`qa_vorticity_screen_20261020.json` records the completed native FAX CPU
comparisons. Its gates fail; it is not a validated AE3D-equivalent answer.
See `docs/qa_ae3d_vorticity_benchmark.rst` for assumptions and reproduction.

`qa_vorticity_axis_20261020.json` records the full-domain follow-up and a
control that changes only the m=0 axis condition. That change shifts the
selected frequency by 0.0183%; the large QA discrepancy remains. It also
records the numerical equivalence and measured speedup of batched geometry
reconstruction. These are completed native matrix calculations, without
new particle tracing or changes to the archived FIRM3D loss result.

`qa_vorticity_operator_audit_20261020.json` records the subsequent independent
Cartesian check and controlled source-formula audit. Two magnetic-shear
mode-index interchanges reproduce nearly all of the public-AE3D-formula
improvement; reversing the prefactor derivatives alone does not. The native
FAX bending equations remain unchanged. See `docs/qa_ae3d_operator_audit.rst`
for the product-rule derivation and limits of the comparison.

`qa_ae3d_harmonic_header.json` records the actual 162-harmonic QA basis,
49 radial coordinates and source provenance from the approved bounded
archive-header read on 2026-10-21. It differs from the paper table's 158
harmonics; the eigenvalue and radial grid match the tracing export.
Use `--harmonics archive` for new native comparisons to that run.

`qa_vorticity_radial_audit_20261021.json` records the subsequent sampling,
quadrature, actual-basis, endpoint-convention, Cartesian and radial-refinement
controls. Its NPZ companion stores only numeric signed profiles and JSON
metadata so the figure can be regenerated with
`python examples/research/plot_qa_ae3d_radial_audit.py`, without running JAX
or accessing the network. Reference data retain the CC BY 4.0 attribution
above. The reconstructed public-source mode reaches 4.3% profile error when
the inspected endpoint convention is reproduced diagnostically; the native
FAX QA eigenfunction is still not radially converged. No production bending
equations or saved ESSOS loss results were changed by this audit.
