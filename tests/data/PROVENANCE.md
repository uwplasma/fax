# Fixture provenance

Every external reference in `tests/data` is listed here: where it comes from, how to verify it, which model switches and
units apply, and which test pins it with which tolerance. Frequencies are `f = omega / 2 pi` in kHz unless stated.

## `knyazev_qa.npz` (0.67 MB): precise QA, STELLGAP continuum and AE3D band

| Item | Value |
|---|---|
| Identifier | Knyazev et al., "On shear Alfvén wave-induced energetic ion transport in optimized stellarators", *J. Plasma Phys.* (2026), arXiv:2603.03118. Data archive "Data and figure-generating scripts for: ...", Zenodo concept DOI 10.5281/zenodo.21299907, version DOI 10.5281/zenodo.21299908 |
| URL | <https://zenodo.org/records/21299908>, file `Knyazev_JPP2026.zip` (1 127 897 428 bytes) |
| License | CC-BY-4.0 (Zenodo record metadata). Attribution: Knyazev et al. (2026) |
| SHA-256 of the archive | `615ceea03bc96fa5687728b3fe3e10c62c3d2594d4d4d02bdeeef26fbb1e76f4` (MD5 `7f65aebb55500e7bba026fd4fd91510b`, as listed by Zenodo) |
| Members read (SHA-256) | `simulations_fig4_fig5/QA/all_inputs/boozmn_betaQA.nc` `6a32726c66b57dc5d68e31bbf53f6f5833f1e7fb124404dadfe76873beb2b6d5`; `FIG2-STELLGAP_and_AE3D_for_QI_and_QA/inputs/QA/stellgap/alfven_spec{0,1,2,3}` `cd4ef4aa19669e1a123464eea1f55632bbd04a5b6beeae1709c5ae2f85e8d212`, `9f1aafaeb0660d5e5bd84862cb698eeb03d6d49cdad827058551d86113f039a2`, `c6e2add386c7dc8a374ade1dea07d968e016871e86491c26870c191d197c5274`, `a2946c52fad44a94e11cb2660139d1a1ec2e915c77a009f745887fcda3024385`; `.../QA/ae3d/egn_mode_asci.dat` `78086943957525a6641578ee86ac5b289788af571cd476ae35359637d7d09511` |
| Model switches | Precise QA, `nfp = 2`, finite-beta VMEC equilibrium (`ns = 51`, aspect 6, `p(0) = 7.1e5` Pa) transformed by `booz_xform` (`mboz = 96`, `nboz = 23`). AE3D with its kink term removed (archive README); no pressure-curvature term either, so FAX runs with `finite_beta=False`. STELLGAP: the 162 branches of the same harmonic list. Sine basis `Phi = sum Phi_mn sin(m theta - n zeta)` in AE3D |
| Units, normalization | Flat mass density 1.88e-6 kg/m^3: the archive does not contain the STELLGAP/AE3D decks, so the value is inferred (it reproduces the STELLGAP continuum to 3e-3 and agrees with the archived AE3D frequency scale; the same value is used in PR #9). STELLGAP `omega^2 = alpha / beta` (columns 2 and 4 of `alfven_spec`, kHz^2); AE3D eigenvalue `= f^2` in kHz^2 (negative for unstable); AE3D eigenvectors are potential amplitudes at the nodes with arbitrary scale. STELLGAP labels surface `k` as `k / (ns - 1)`, one cell (0.02) outward of the physical `(k - 1) / (ns - 1)`, as in CTOK |
| Resolution | 49 radial nodes `s = 0.02, ..., 0.98` (the full mesh), 162 harmonics `(m, n)`, `m <= 26`, `n` from -2 to 16 in steps of 2 (full-torus `n`, `nfp = 2`). Stored: Boozer harmonics above 1e-5 of the largest (198 of 4489, float32); 98 of 784 STELLGAP surfaces; all 7938 AE3D frequencies; the 20 radially extended gap modes in 26-35 kHz (selection rule in `examples/knyazev_qa.py`) on their 96 largest harmonics, and the archived global mode (30.3086 kHz) on all 162 |
| Generation command | `python examples/knyazev_qa.py` with `ARCHIVE` set and `REBUILD = True` (prints the hashes above) |
| Pinned by | `tests/test_knyazev.py`; tolerances and their convergence budget are in `docs/validation.md` ("Independent 3-D reference") |

## `boozmn_ctok.nc`, `stellgap_ctok_continuum.npz`, `ae3d_ctok_modes.npz`: CTOK tokamak

| Item | Value |
|---|---|
| Identifier | Public CTOK test problems of STELLGAP and AE3D (circular tokamak, aspect 3.33, `ns = 101`, `nfp = 1`, `mboz = 48`, `nboz = 1`, 143 harmonics) |
| URL | <https://github.com/ORNL-Fusion/Stellgap>, <https://github.com/ORNL-Fusion/AE3D> (released `*_test_problem*.tar.gz` packets) |
| License | MIT (both repositories) |
| SHA-256 | `boozmn_ctok.nc` `63d3005cb8796ac64698405ea485b422faf6fc1224bdc84411a7d8584ec43403`; `stellgap_ctok_continuum.npz` `ff131c776a595c4830eecf7bfbd72b483e305c1e857b9dfa859fc545c12c1f60`; `ae3d_ctok_modes.npz` `b6e54a1d79e73bf614b0663f4bf4ecbd1559d2918549051274c1982b43359155` |
| Model switches | `n = 2`, `m = 0..12`, deuterium 1e19 m^-3 (`rho = 3.34e-8` kg/m^3, flat); AE3D without pressure and `J_par` terms (`finite_beta=False`) |
| Units, normalization | kHz, `f = omega / 2 pi`. STELLGAP file: 296 surfaces `s = 0.02 ... 0.9869` and the 13 sorted branches in kHz (STELLGAP labels shifted one cell, see `test_stellgap_ctok_parity`). AE3D file: 7 eigenvalues in kHz (337.92, 343.65, 384.13, 685.59, 703.30, 738.23, 741.21) and their eigenvectors on the 99-point full mesh, 13 harmonics, float32, arbitrary scale |
| Resolution | 99 radial nodes, 13 harmonics |
| Generation command | Copied from the packets by hand in P2 (commit `d57803d`); the reduction of AE3D's `egn_mode_asci.dat` to 7 modes was not scripted and is not reproducible from this repository |
| Pinned by | `tests/test_continuum.py::test_stellgap_ctok_parity` (mean and p95 relative error 7e-3), `tests/test_eigen.py::test_ae3d_ctok_parity` (6 modes within 1.5%, overlap > 0.9; the TAE is the unresolved discrepancy of `docs/validation.md`) |
