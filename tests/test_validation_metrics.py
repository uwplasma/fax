from __future__ import annotations

import numpy as np

from fax.validation import hermitian_relative_error, max_relative_error, observed_order, relative_l2_error


def test_relative_errors_are_zero_for_exact_match():
    actual = np.asarray([1.0, 2.0, 3.0])

    assert float(relative_l2_error(actual, actual)) == 0.0
    assert float(max_relative_error(actual, actual)) == 0.0


def test_hermitian_error_detects_symmetry_breaking():
    hermitian = np.asarray([[1.0, 2.0], [2.0, 3.0]])
    nonhermitian = np.asarray([[1.0, 2.0], [2.5, 3.0]])

    assert float(hermitian_relative_error(hermitian)) == 0.0
    assert float(hermitian_relative_error(nonhermitian)) > 0.0


def test_observed_order_second_order_case():
    order = observed_order(4.0e-4, 1.0e-4, refinement=2.0)

    np.testing.assert_allclose(order, 2.0)

