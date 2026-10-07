import pytest

from vjaudit.report.metrics import cohen_kappa, error_rates, yes_rate


def test_kappa_and_rates_hand_values():
    j, r = [True, True, False, True], [True, False, False, True]
    assert cohen_kappa(j, r) == pytest.approx(0.5)
    assert error_rates(j, r) == {"fpr": 0.5, "fnr": 0.0, "n_ref_pos": 2, "n_ref_neg": 2}
    assert yes_rate(j) == 0.75


def test_kappa_undefined_when_constant():
    assert cohen_kappa([True] * 3, [True] * 3) is None
    assert cohen_kappa([], []) is None


def test_kappa_perfect_and_opposite():
    assert cohen_kappa([True, False], [True, False]) == pytest.approx(1.0)
    assert cohen_kappa([True, False], [False, True]) == pytest.approx(-1.0)


def test_rates_none_when_denominator_zero():
    assert error_rates([True, False], [True, True]) == {"fpr": None, "fnr": 0.5, "n_ref_pos": 2, "n_ref_neg": 0}
    assert yes_rate([]) is None
