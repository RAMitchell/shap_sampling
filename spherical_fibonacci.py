import numpy as np
from scipy.optimize import root_scalar
from numba import njit
import math


def phi(d):
    x = 2.0
    for i in range(10):
        x = np.power(1 + x, 1 / (d + 1))
    return x


def x_sequence(n, dimension):
    X = np.zeros((n, dimension))
    a = np.zeros(dimension)
    for j in range(1, dimension):
        a[j - 1] = 1 / phi(j) ** j
    a[dimension - 1] = 1 / (n + 1)

    for i in range(n):
        for j in range(dimension - 1):
            X[i][j] = ((i + 1) * a[j]) % 1.0
        X[i][dimension - 1] = ((i + 0.5) * a[dimension - 1]) % 1.0
    return X


def test_x_sequence():
    # integrate a portion of the unit cube
    dimensions = [2, 3, 5, 10]
    n = 10000
    for d in dimensions:
        X = x_sequence(n, d)
        est = 0.0
        for i in range(n):
            est += np.all(X[i] > 0.3)
        est /= n
        assert np.isclose(0.7 ** d, est, atol=1e-3)


@njit
def integrate_sin(x, m):
    if m == 0:
        return x
    elif m == 1:
        return 1 - np.cos(x)
    else:
        return (m - 1) / m * integrate_sin(x, m - 2) - np.cos(x) * np.sin(x) ** (
                m - 1
        ) / m


# 1 based indexing
@njit
def polar_cdf(index, varphi, d):
    assert index > 0
    if index == d - 1:
        return varphi / (2 * np.pi)
    beta_inv = (1 / np.sqrt(np.pi)) * (
            math.gamma(((d - index) + 1) / 2) / math.gamma((d - index) / 2))
    return beta_inv * integrate_sin(varphi, d - index - 1)


def test_polar_cdf():
    d = 10
    for i in range(1, d - 1):
        assert np.isclose(polar_cdf(i, 0, d), 0.0)
        assert np.isclose(polar_cdf(i, np.pi, d), 1.0)
    assert np.isclose(polar_cdf(d - 1, 0, d), 0.0)
    assert np.isclose(polar_cdf(d - 1, 2 * np.pi, d), 1.0)


# 1 based indexing
def inverse_polar_cdf(x, d, index):
    root_function = lambda varphi: polar_cdf(index, varphi, d) - x
    upper_bracket = np.pi if index < d - 1 else 2 * np.pi
    bracket = [0, upper_bracket]
    result = root_scalar(root_function, bracket=bracket, xtol=1e-5, rtol=1e-5)
    return result.root


def test_inverse_polar_cdf():
    d = 10
    for i in range(1, d - 1):
        a = inverse_polar_cdf(0.0, d, i)
        b = inverse_polar_cdf(1.0, d, i)
        assert np.isclose(a, 0.0, rtol=1e-3, atol=1e-3)
        assert np.isclose(b, np.pi, rtol=1e-3, atol=1e-3)
    assert np.isclose(inverse_polar_cdf(0.0, d, d - 1), 0.0, rtol=1e-3, atol=1e-3)
    assert np.isclose(inverse_polar_cdf(1.0, d, d - 1), np.pi * 2, rtol=1e-3, atol=1e-3)


def polar_to_real(polar, d, r=1.0):
    real = np.full(d, r)
    for i in range(d):
        for j in range(i):
            real[i] *= np.sin(polar[j])
        if i < d - 1:
            real[i] *= np.cos(polar[i])
    return real


def fibonacci_points(n, d):
    Y = np.zeros((n, d))
    X = x_sequence(n, d - 1)
    for i in range(n):
        polar = np.zeros(d - 1)
        for j in range(0, d - 1):
            polar[j] = inverse_polar_cdf(X[i][j], d, j + 1)
        Y[i] = polar_to_real(polar, d)
    return Y


def test_fibonacci_points():
    X = fibonacci_points(10000, 20)
    for x in X:
        assert np.isclose(1.0, np.linalg.norm(x))
    counts = np.zeros(X.shape[1])
    for x in X:
        counts += x > 0.0
    counts /= X.shape[0]
    assert np.allclose(counts, 0.5, rtol=1e-2, atol=1e-2)

    # Plot the distributions of each coordinate
    # They should look normal
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 4)
    for j in range(X.shape[1]):
        axes[j // 4, j % 4].hist(X[:, j], bins=20)
    plt.show()

    X = fibonacci_points(100, 3)
    assert nearest_neighbour(X) > 0.2


def nearest_neighbour(X):
    n = X.shape[0]
    gram = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            gram[i, j] = np.linalg.norm(X[i] - X[j])
    lower_triangular = gram[np.tril_indices_from(gram, k=-1)]
    return np.min(lower_triangular)


def zero_sum_projection(d):
    basis = np.array([[1.0] * i + [-i] + [0.0] * (d - i - 1) for i in range(1, d)])
    return np.array([v / np.linalg.norm(v) for v in basis])


def fibonacci_permutations(n, d):
    sphere_points = fibonacci_points(n, d - 1)
    basis = zero_sum_projection(d)
    projected_sphere_points = sphere_points.dot(basis)
    p = np.zeros((n, d), dtype=np.int64)
    for i in range(n):
        p[i] = np.argsort(projected_sphere_points[i])
    return p


def plot_permutation_histogram():
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.style.use("seaborn")
    plt.rc('font', family='serif')
    p = fibonacci_permutations(1000, 4)
    strings = []
    for x in p:
        strings.append(str(np.argsort(x)))
    strings = sorted(strings)
    sns.histplot(data=strings)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


plot_permutation_histogram()
