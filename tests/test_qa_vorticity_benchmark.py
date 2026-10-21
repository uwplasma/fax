"""Protect physical-profile matching and the published benchmark inputs."""
import json

import numpy as np
import pytest

from examples.research.qa_ae3d_vorticity_benchmark import (
    REFERENCE, HARMONIC_HEADER, alignment, archived_qa_harmonics, interpolate_profiles,
    paper_qa_harmonics, reference_indices,
)


def test_published_basis_contains_the_retained_tracing_harmonics():
    labels = paper_qa_harmonics()
    assert labels.shape == (158, 2)
    assert len({tuple(row) for row in labels}) == 158
    with np.load(REFERENCE, allow_pickle=False) as data:
        indices = reference_indices(labels, data['harmonics'], 'cosine')
        assert len(indices) == 35
        metadata = json.loads(str(data['metadata_json']))
        # Preserve the tracing input and native-wave frequency as distinct data.
        assert metadata['archive_eigenvalue_sqrt_times_1000'] == pytest.approx(30308.607688245924)
        assert metadata['archived_transport_omega_rad_s']/(2*np.pi) == pytest.approx(4823.764731817361)
        assert metadata['source_sha256'] == '17983671ea72b829066b139d20d90127e696a5e378ed1c326b2e8f3584fecc36'


def test_global_phase_alignment_does_not_hide_relative_harmonic_errors():
    s = np.linspace(.1, .9, 9)
    reference = np.column_stack([s*(1-s), .4*s*s])
    same = (2+3j)*reference
    different = same.copy()
    different[:, 1] *= -1
    profiles = np.stack([same, different])
    gain, overlap, error = alignment(profiles, reference, s)
    np.testing.assert_allclose(gain[0]*same, reference, atol=1e-14)
    assert overlap[0] == pytest.approx(1.)
    assert error[0] < 2e-8
    assert overlap[1] < .9 and error[1] > .4


def test_actual_archive_basis_and_table_differences_are_preserved():
    labels = archived_qa_harmonics()
    actual, table = set(map(tuple, labels)), set(map(tuple, paper_qa_harmonics()))
    assert labels.shape == (162, 2)
    assert actual-table == {(0, 4), (1, 4), (2, 4), (3, 4), (24, 16), (25, 16), (26, 16)}
    assert table-actual == {(24, 20), (25, 20), (26, 20)}
    with np.load(REFERENCE, allow_pickle=False) as data:
        assert len(reference_indices(labels, data['harmonics'], 'cosine')) == 35
    header = json.loads(HARMONIC_HEADER.read_text())
    assert header['matrix_dimension'] == 7938
    assert header['radial_points'] == 49
    assert header['cached_payload_bytes'] <= 64*1024
    assert header['reference_radial_grid_max_absolute_difference'] == 0.
    assert header['matching_reference_eigenvalue_relative_difference'] < 1e-13


def test_profile_comparison_evaluates_piecewise_linear_elements_without_smoothing():
    nodes = np.array([.1, .3, .6, .9])
    values = np.array([0., 1., -.5, 0.])[None, :, None]
    samples = np.array([.15, .45, .75])
    np.testing.assert_allclose(interpolate_profiles(nodes, values, samples).ravel(), [.25, .25, -.25], atol=1e-14)
