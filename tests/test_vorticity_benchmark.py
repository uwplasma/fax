"""One small refinement test checks recovery of the prescribed exact solution."""
import numpy as np

from examples.research.vorticity_benchmark import benchmark_case


def test_exact_manufactured_solution_distinguishes_both_shear_index_changes():
    cases = [benchmark_case(n) for n in (9, 17, 33)]
    native = np.array([rows[0]['l2_error'] for rows in cases])
    assert np.all(np.log2(native[:-1]/native[1:]) > 1.9)
    assert native[-1] < .003
    assert cases[-1][0]['weak_residual'] < 2e-5
    for index in (1, 2, 3):
        assert cases[-1][index]['l2_error'] > .015
        assert cases[-1][index]['l2_error'] > .5*cases[0][index]['l2_error']
        assert cases[-1][index]['weak_residual'] > .005
    zero = benchmark_case(9, shear=0)
    for row in zero[1:]:
        for metric in ('l2_error', 'energy_error', 'weak_residual'):
            assert row[metric] == zero[0][metric]
