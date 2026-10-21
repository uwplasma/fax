"""Native FAX QA vorticity benchmark against the published AE3D mode.

No imported matrices, particle tracing, or downloads. CPU by default, with a
120-second process cap. Each completed case is saved before starting the next.
The reference stores the 35 physical-potential harmonics retained for tracing;
Use --harmonics archive for the actual 162-harmonic archived solve. The paper
table's differing 158-harmonic basis remains available with --harmonics paper.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np


REFERENCE = Path(__file__).with_name('references')/'qa_ae3d_potential.npz'
HARMONIC_HEADER = REFERENCE.with_name('qa_ae3d_harmonic_header.json')
FREQUENCY_SOURCE = 'https://github.com/ORNL-Fusion/AE3D/blob/master/ae_solve_version31.f'


def paper_qa_harmonics():
    """Cosine labels in Knyazev et al. appendix A, table 2; no inferred ranges."""
    ranges = [(-2, 1, 11), (0, 1, 20), (2, 0, 26), (4, 4, 26), (6, 4, 26),
              (8, 8, 26), (10, 12, 26), (12, 17, 26), (14, 20, 26), (20, 24, 26)]
    return np.array([(m, n) for n, first, last in ranges for m in range(first, last+1)], dtype=int)


def archived_qa_harmonics():
    """Actual cosine labels read from the archived QA eigenmode header."""
    header = json.loads(HARMONIC_HEADER.read_text())
    labels = np.asarray(header['harmonics'], dtype=int)
    if (labels.shape != (header['harmonic_count'], 2)
            or header['matrix_dimension'] != header['radial_points']*len(labels)
            or len({tuple(row) for row in labels}) != len(labels)):
        raise ValueError('Inconsistent archived QA harmonic header')
    return labels


def radial_weights(s):
    return np.r_[np.diff(s)[0]/2, (s[2:]-s[:-2])/2, np.diff(s)[-1]/2]


def interpolate_profiles(s, profiles, target):
    """Evaluate the actual linear finite elements, without adding a spline."""
    index = np.clip(np.searchsorted(s, target, side='right')-1, 0, len(s)-2)
    weight = (target-s[index])/(s[index+1]-s[index])
    return profiles[..., index, :]*(1-weight[:, None])+profiles[..., index+1, :]*weight[:, None]


def alignment(profiles, reference, s):
    weights = radial_weights(s)
    norm = np.sum(weights[:, None]*abs(profiles)**2, axis=(-2, -1))
    ref_norm = np.sum(weights[:, None]*abs(reference)**2)
    cross = np.sum(weights[:, None]*profiles.conj()*reference, axis=(-2, -1))
    gain = cross/np.maximum(norm, 1e-300)
    overlap = np.clip(abs(cross)/np.sqrt(np.maximum(norm*ref_norm, 1e-300)), 0, 1)
    return gain, overlap, np.sqrt(np.maximum(1-overlap**2, 0))


def reference_indices(labels, reference_labels, basis):
    lookup = {tuple(row): i for i, row in enumerate(labels)}
    indices = []
    for row in reference_labels:
        index = lookup.get(tuple(row))
        if index is None and basis == 'cosine':
            index = lookup.get(tuple(-row))
        if index is None:
            raise ValueError(f'Reference harmonic {row} is absent from the solve basis')
        indices.append(index)
    return np.array(indices)


def run_case(geometry, labels, ref, metadata, radial_points, angles, basis, directory, radial_domain='axis'):
    import jax
    import jax.numpy as jnp
    from fax.boozer import slice_boozer_geometry_surfaces
    from fax.vorticity import physical_potential_problem_from_boozer_geometry, solve_physical_potential_eigenmodes

    if radial_domain != 'annulus':
        geo = geometry  # Never coarsen the background when refining Phi.
        options = dict(radial_nodes=np.linspace(0., 1., radial_points),
                       boundary_condition='axis' if radial_domain == 'axis' else 'dirichlet')
    else:
        indices = np.unique(np.rint(np.linspace(0, len(geometry.s)-1, radial_points)).astype(int))
        geo, options = slice_boozer_geometry_surfaces(geometry, indices), {}
    density = float(metadata['mass_density_kg_m3'])
    build = jax.jit(lambda g, rho: physical_potential_problem_from_boozer_geometry(g, labels,
        mass_density=rho, theta_points=angles, zeta_points=angles, angular_basis=basis, **options))
    started = time.perf_counter()
    problem = jax.block_until_ready(build(geo, density))
    cold = time.perf_counter()-started
    started = time.perf_counter()
    problem = jax.block_until_ready(build(geo, density))
    warm = time.perf_counter()-started
    started = time.perf_counter()
    result = solve_physical_potential_eigenmodes(problem)
    jax.block_until_ready(result.omega2)
    solve_seconds = time.perf_counter()-started
    s = np.asarray(problem.s)
    structures = np.asarray(result.mode_structures)
    profiles = interpolate_profiles(s, structures, ref['s'])
    selected_columns = reference_indices(labels, ref['harmonics'], basis)
    comparison = profiles[..., selected_columns]
    gain, overlap, error = alignment(comparison, ref['potential_volts'], ref['s'])
    frequencies = np.asarray(result.frequencies_hz)
    residuals = np.asarray(result.residuals)
    valid = np.isfinite(frequencies) & (frequencies > 0) & np.isfinite(residuals) & (residuals < 1e-8)
    if not np.any(valid):
        raise ValueError('No positive native eigenpair passes the residual gate')
    best = int(np.argmax(np.where(valid, overlap, -1)))
    reference_hz = float(metadata['archive_eigenvalue_sqrt_times_1000'])
    nearest = int(np.argmin(np.where(valid, abs(frequencies-reference_hz), np.inf)))
    total_norm = np.sum(radial_weights(ref['s'])[:, None]*abs(profiles[best])**2)
    retained_norm = np.sum(radial_weights(ref['s'])[:, None]*abs(comparison[best])**2)
    reference_nodes = np.column_stack([np.interp(s, ref['s'], ref['potential_volts'][:, j])
                                      for j in range(len(ref['harmonics']))])
    projected = np.zeros((len(s), len(labels)))
    projected[:, selected_columns] = reference_nodes
    vector = projected.ravel()[np.asarray(problem.active_dofs)]
    rayleigh = np.vdot(vector, np.asarray(problem.stiffness)@vector).real/np.vdot(vector, np.asarray(problem.mass)@vector).real
    frequency_error = abs(frequencies[best]/reference_hz-1)
    summary = dict(radial_points=len(s), angular_points=angles, harmonics=len(labels), angular_basis=basis,
        radial_domain=radial_domain, boundary_condition=dict(problem.metadata)['boundary_condition'],
        matrix_dimension=len(frequencies), boundary_s=s[[0, -1]].tolist(),
        selected_index=best, selection='largest overlap with the retained signed physical Phi profiles',
        frequency_hz=float(frequencies[best]), frequency_relative_error=float(frequency_error),
        retained_profile_overlap=float(overlap[best]), retained_profile_relative_l2=float(error[best]),
        native_potential_norm_fraction_outside_reference=float(max(1-retained_norm/total_norm, 0)),
        residual=float(residuals[best]), max_spectrum_residual=float(residuals.max()),
        gain_real=float(gain[best].real), gain_imag=float(gain[best].imag),
        nearest_frequency_mode=dict(index=nearest, frequency_hz=float(frequencies[nearest]),
            retained_profile_overlap=float(overlap[nearest])),
        projected_reference_rayleigh_frequency_hz=float(np.sqrt(rayleigh)/(2*np.pi)),
        assembly_cold_seconds=cold, assembly_warm_seconds=warm, solve_seconds=solve_seconds,
        retained_data_gates_passed=bool(frequency_error <= .01 and error[best] <= .1 and residuals[best] <= 1e-8),
        claim='Retained-harmonic comparison; exact AE3D discretization parity not certified')
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory/'mode.npz', s=s, harmonics=labels, structure=structures[best],
        frequency_hz=frequencies[best], angular_basis=basis, reference_s=ref['s'],
        aligned_retained_profiles=gain[best]*comparison[best])
    np.savez_compressed(directory/'spectrum.npz', frequencies_hz=frequencies, residuals=residuals,
                        retained_profile_overlap=overlap, retained_profile_relative_l2=error)
    (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary, gain[best]*comparison[best]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--boozmn', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--cases', nargs='+', default=['8:64', '12:64', '16:64'], help='radial:angular points')
    parser.add_argument('--harmonics', choices=['retained', 'paper', 'archive'], default='retained')
    parser.add_argument('--basis', choices=['cosine', 'complex'], default='cosine')
    parser.add_argument('--radial-domain', choices=['axis', 'full-dirichlet', 'annulus'], default='axis',
        help='Regular full domain (default), same geometry with zero axis, or historical annular screen')
    parser.add_argument('--platform', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--wall-seconds', type=float, default=120.)
    parser.add_argument('--output', type=Path, default=Path('outputs/qa_ae3d_vorticity_benchmark'))
    args = parser.parse_args()
    cases = [tuple(map(int, case.split(':'))) for case in args.cases]
    if (any(len(case) != 2 or case[0] < 4 or case[1] < 8 for case in cases)
            or not np.isfinite(args.wall_seconds) or args.wall_seconds <= 0):
        parser.error('Need radial:angular cases with at least 4:8 points and a positive wall cap')
    if args.harmonics in ('paper', 'archive') and args.basis != 'cosine':
        parser.error('Paper/archive labels specify the cosine sector; use --basis cosine')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose a fresh output directory to preserve completed data')
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(JAX_PLATFORMS=args.platform, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    report = dict(status='running', cases=[], particle_tracing=False, imported_matrices=False,
                  paper='https://arxiv.org/html/2603.03118v1', frequency_source=FREQUENCY_SOURCE,
                  frequency_convention='AE3D scales inertia by (1000*2*pi)^2; sqrt(eigenvalue) is kHz',
                  acceptance=dict(frequency_relative_error=.01, retained_profile_relative_l2=.1, residual=1e-8))

    def save():
        temporary = args.output/'summary.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(args.output/'summary.json')

    def stop():
        report['status'] = 'wall_clock_limit_reached'
        save()
        print('Wall cap reached; completed cases saved', flush=True)
        os._exit(124)

    timer = threading.Timer(args.wall_seconds, stop)
    timer.daemon = True
    timer.start()
    started = time.perf_counter()
    try:
        import jax
        import jax.numpy as jnp
        import fax.vorticity
        from netCDF4 import Dataset
        from fax.boozer import read_boozmn_geometry
        jax.config.update('jax_enable_x64', True)
        with np.load(args.reference, allow_pickle=False) as data:
            ref = {key: data[key] for key in ('s', 'harmonics', 'potential_volts')}
            metadata = json.loads(str(data['metadata_json']))
        digest = hashlib.sha256(args.boozmn.read_bytes()).hexdigest()
        if digest != metadata['equilibrium_sha256']:
            raise ValueError('Equilibrium hash differs from the published QA reference')
        geometry = read_boozmn_geometry(args.boozmn)
        with Dataset(args.boozmn) as nc:
            actual_s = (np.asarray(nc['jlist'][:], float)-1.5)/(int(nc['ns_b'][:])-1)
        geometry = replace(geometry, s=jnp.asarray(actual_s))
        labels = (ref['harmonics'] if args.harmonics == 'retained' else
                  archived_qa_harmonics() if args.harmonics == 'archive' else paper_qa_harmonics())
        if args.radial_domain == 'annulus' and any(case[0] > len(actual_s) for case in cases):
            raise ValueError('Requested more radial nodes than the stored geometry has')
        report.update(reference=metadata, platform=jax.default_backend(), device_kind=jax.devices()[0].device_kind,
            operator_sha256=hashlib.sha256(Path(fax.vorticity.__file__).read_bytes()).hexdigest(),
            benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            radial_coordinate='Fortran boozmn half mesh: (jlist-1.5)/(ns_b-1)',
            radial_domain=args.radial_domain, harmonic_source=args.harmonics,
            harmonic_header_sha256=(hashlib.sha256(HARMONIC_HEADER.read_bytes()).hexdigest()
                                    if args.harmonics == 'archive' else None),
            boundary_policy=dict(axis='s=0..1; regular m=0 axis, other m zero; Phi=0 at LCFS; interior geometry quadrature',
                **{'full-dirichlet': 's=0..1; all axis/edge values zero; interior geometry quadrature',
                   'annulus': 'Phi=0 at first and last half-mesh nodes'})[args.radial_domain])
        save()
        previous = None
        for radial, angular in cases:
            name = f'{radial}_{angular}_{args.basis}_{len(labels)}_{args.radial_domain}'
            print(f'Native FAX {name}: assembling and solving', flush=True)
            summary, profile = run_case(geometry, labels, ref, metadata, radial, angular, args.basis,
                args.output/name, args.radial_domain)
            if previous is not None:
                _, ov, err = alignment(profile, previous[1], ref['s'])
                summary['change_from_previous_case'] = dict(
                    case=previous[0]['case'], frequency_relative_change=abs(summary['frequency_hz']/previous[0]['frequency_hz']-1),
                    retained_profile_overlap=float(ov), retained_profile_relative_l2=float(err))
            summary['case'] = name
            report['cases'].append(summary)
            previous = summary, profile
            save()
            print(f"  {summary['frequency_hz']/1000:.6f} kHz; overlap={summary['retained_profile_overlap']:.6f}; "
                  f"profile error={summary['retained_profile_relative_l2']:.6f}; residual={summary['residual']:.3g}", flush=True)
        report.update(status='completed', elapsed_seconds=time.perf_counter()-started,
                      retained_data_gates_passed=report['cases'][-1]['retained_data_gates_passed'],
                      full_ae3d_parity=False)
        save()
    except Exception as error:
        report.update(status='failed', error=str(error))
        save()
        raise
    finally:
        timer.cancel()


if __name__ == '__main__':
    main()
