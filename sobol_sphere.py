import numpy as np
from scipy.optimize import root_scalar
from scipy.special import beta
from numba import njit


@njit
def int_sin_m(x, m):
    if m == 0:
        return x
    elif m == 1:
        return 1 - np.cos(x)
    else:
        return (m - 1) / m * int_sin_m(x, m - 2) - np.cos(x) * np.sin(x) ** (
                m - 1
        ) / m


def equal_area_projection(X, d):
    n = X.shape[0]
    Y = np.ones((n, d))
    for i in range(n):
        Y[i][0] *= np.sin(X[i, 0] * 2 * np.pi)
        Y[i][1] *= np.cos(X[i, 0] * 2 * np.pi)

    for j in range(2, d):
        inv_beta = 1 / beta(j / 2, 1 / 2)
        for i in range(n):
            root_function = lambda varphi: inv_beta * int_sin_m(varphi, j - 1) - X[i, j - 1]
            deg = root_scalar(root_function, bracket=[0, np.pi], xtol=1e-15).root
            for k in range(j):
                Y[i][k] *= np.sin(deg)
            Y[i][j] *= np.cos(deg)
    return Y


def sobol_sphere(n, d):
    import qmcpy
    X = qmcpy.Sobol(d - 1, graycode=True).gen_samples(n)
    return equal_area_projection(X, d)


def test_points_distribution():
    X = sobol_sphere(10000, 20)
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


def zero_sum_projection(d):
    basis = np.array([[1.0] * i + [-i] + [0.0] * (d - i - 1) for i in range(1, d)])
    return np.array([v / np.linalg.norm(v) for v in basis])


def sobol_permutations(n, d):
    sphere_points = sobol_sphere(n, d - 1)
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
    p = sobol_permutations(1000, 4)
    strings = []
    for x in p:
        strings.append(str(np.argsort(x)))
    strings = sorted(strings)
    plt.figure(figsize=(5.2, 3.62))
    sns.histplot(data=strings)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("figures/sobol_histogram.png", dpi=200)
    plt.show()

# plot_permutation_histogram()

