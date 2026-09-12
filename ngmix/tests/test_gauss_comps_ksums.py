"""
the closed-form model sums kernel: matches a plain python evaluation
for a valid covariance, and returns nan sums rather than raising for
singular, numerically singular or non positive definite totals
"""
import numpy as np
import pytest

from ngmix.prepsfadmom.models_nb import gauss_comps_ksums, DET_REL_TOL


def python_sums(F, cov_sm, dv, du, wt_cov, detAtinv):
    """
    the same sums in plain numpy: for each component the product
    gaussian of the weight and the component
    """
    sums = np.zeros(6)
    for k in range(len(F)):
        C = wt_cov + cov_sm[k]
        Ci = np.linalg.inv(C)
        d = np.array([dv[k], du[k]])
        Cd = Ci @ d
        sflux = F[k] * np.exp(-0.5 * d @ Cd) / (
            2 * np.pi * detAtinv * np.sqrt(np.linalg.det(C))
        )
        mu = wt_cov @ Cd
        Sp = wt_cov @ Ci @ cov_sm[k]
        vv = Sp[0, 0] + mu[0] ** 2
        vu = Sp[0, 1] + mu[0] * mu[1]
        uu = Sp[1, 1] + mu[1] ** 2
        sums += sflux * np.array([
            mu[0], mu[1], uu - vv, 2 * vu, uu + vv, 1.0,
        ])
    return sums


def run_kernel(F, cov_sm, dv, du, wt_cov, detAtinv=1.0):
    sums = np.zeros(6)
    gauss_comps_ksums(
        np.asarray(F, dtype='f8'),
        np.array([s[0, 0] for s in cov_sm]),
        np.array([s[0, 1] for s in cov_sm]),
        np.array([s[1, 1] for s in cov_sm]),
        np.asarray(dv, dtype='f8'), np.asarray(du, dtype='f8'),
        wt_cov[0, 0], wt_cov[0, 1], wt_cov[1, 1], detAtinv, sums,
    )
    return sums


def test_matches_python():
    wt_cov = np.array([[0.8, 0.1], [0.1, 0.6]])
    cov_sm = [
        np.array([[0.3, -0.05], [-0.05, 0.4]]),
        np.array([[1.5, 0.2], [0.2, 1.1]]),
    ]
    F = [2.0, 5.0]
    dv = [0.1, -0.3]
    du = [-0.2, 0.5]
    sums = run_kernel(F, cov_sm, dv, du, wt_cov, detAtinv=0.9)
    expected = python_sums(F, cov_sm, dv, du, wt_cov, 0.9)
    assert np.allclose(sums, expected, rtol=1e-12)


@pytest.mark.parametrize('case', ['zero', 'singular', 'negative', 'rounding'])
def test_bad_covariance_gives_nan(case):
    if case == 'zero':
        # both the weight and the component are zero
        wt_cov = np.zeros((2, 2))
        cov_sm = [np.zeros((2, 2))]
    elif case == 'singular':
        # rank one total covariance, exact zero determinant
        wt_cov = np.array([[1.0, 1.0], [1.0, 1.0]])
        cov_sm = [np.zeros((2, 2))]
    elif case == 'negative':
        wt_cov = np.array([[-1.0, 0.0], [0.0, -1.0]])
        cov_sm = [np.array([[0.5, 0.0], [0.0, 0.5]])]
    else:
        # a runaway weight from a real deblend: the determinant is
        # a difference of two products of order 1e35, and comes out
        # as rounding noise (multiples of 2**65, including zero)
        wt_cov = np.array([
            [1.108013293884106e+20, -4.688552585113404e+17],
            [-4.688552585113404e+17, 1983958627997551.5],
        ])
        cov_sm = [np.array([[53028.737, -225.6145], [-225.6145, 1.2295]])]
        C = wt_cov + cov_sm[0]
        det = C[0, 0] * C[1, 1] - C[0, 1] ** 2
        assert det <= DET_REL_TOL * C[0, 0] * C[1, 1]

    sums = run_kernel([1.0], cov_sm, [0.0], [0.0], wt_cov)
    assert np.all(np.isnan(sums))


def test_good_after_bad_component():
    # a bad component poisons the whole call, as before: the caller
    # treats nan sums as a failed evaluation
    wt_cov = np.array([[0.8, 0.0], [0.0, 0.8]])
    cov_sm = [np.array([[0.3, 0.0], [0.0, 0.3]]), np.zeros((2, 2)) - wt_cov]
    sums = run_kernel([1.0, 1.0], cov_sm, [0.0, 0.0], [0.0, 0.0], wt_cov)
    assert np.all(np.isnan(sums))
