import numpy as np
import ngmix


def test_get_ratio_err_scalar():
    a = 3
    b = 5
    var_a = 1
    var_b = 3
    cov_ab = 0.1

    var = ngmix.util.get_ratio_var(
        a=a, b=b, var_a=var_a, var_b=var_b, cov_ab=cov_ab,
    )
    err = ngmix.util.get_ratio_error(
        a=a, b=b, var_a=var_a, var_b=var_b, cov_ab=cov_ab,
    )

    assert var == (a/b)**2 * (var_a/a**2 + var_b/b**2 - 2*cov_ab/(a*b))
    assert np.allclose(err, np.sqrt(var))
    print(f'var: {var} err: {err}')


def test_get_ratio_err_array():

    a = np.array([3, 5, 6])
    b = np.array([5, 9, 12])

    var_a = np.array([1, 1.2, 0.9])
    var_b = np.array([2, 3, 4])
    cov_ab = np.array([0.1, 0.2, 0.3])

    var = ngmix.util.get_ratio_var(
        a=a, b=b, var_a=var_a, var_b=var_b, cov_ab=cov_ab,
    )
    err = ngmix.util.get_ratio_error(
        a=a, b=b, var_a=var_a, var_b=var_b, cov_ab=cov_ab,
    )

    assert np.allclose(
        var,
        (a/b)**2 * (var_a/a**2 + var_b/b**2 - 2*cov_ab/(a*b))
    )
    assert np.allclose(err, np.sqrt(var))
    print(f'var: {var} err: {err}')


if __name__ == '__main__':
    test_get_ratio_err_scalar()
    test_get_ratio_err_array()
