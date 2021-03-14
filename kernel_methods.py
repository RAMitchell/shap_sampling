from itertools import combinations
import numpy as np
from numba import njit


@njit
def discordant_pairs(a, b):
    n = len(a)
    assert len(b) == n
    discordant = 0
    a = np.argsort(a)
    b = np.argsort(b)
    for i in range(n):
        for j in range(n):
            if (a[i] < a[j] and b[i] > b[j]) or (a[i] > a[j] and b[i] < b[j]):
                discordant += 1
    return discordant


# n^2 implementation
@njit
def kt_kernel(a, b):
    n = len(a)
    return 1.0 - 2 * discordant_pairs(a, b) / (n * (n - 1))


@njit
def mallows_kernel(a, b):
    l = 1.0
    return np.exp(-l * discordant_pairs(a, b))


# Expected value calculated from moment generating function of the number of inversions in a
# permutation
def mallows_expected_value(d, lamb=1.0):
    mgf_expected = 1.0
    for j in range(1, d + 1):
        mgf_expected *= (1 - np.exp(j * -2 * lamb)) / (j * (1 - np.exp(-2 * lamb)))
    return mgf_expected


def test_mallows_expected_value():
    import itertools
    d_tests = [3, 4, 5]
    for d in d_tests:
        p = np.random.permutation(d)
        values = list(map(lambda x: mallows_kernel(p, np.array(x)), itertools.permutations(p)))
        expected = np.mean(values)
        assert np.isclose(expected, mallows_expected_value(d))


def kt_inverse_feature_map(v, n_features):
    inv_p = np.zeros(n_features, dtype=np.int64)
    for i, (j, k) in enumerate(combinations(range(n_features), 2)):
        if v[i] >= 0:
            inv_p[j] += 1
        else:
            inv_p[k] += 1
    p = sorted(np.arange(0, n_features), key=lambda x: inv_p[x])
    return p


@njit
def kt_feature_map(p, n_features):
    p_inv = np.zeros_like(p)
    for i in range(len(p)):
        p_inv[p[i]] = i
    v = np.ones(n_features * (n_features - 1) // 2)
    i = 0
    for j in range(n_features):
        for k in range(j + 1, n_features):
            if p_inv[j] > p_inv[k]:
                v[i] = 1
            else:
                v[i] = -1
            i += 1
    return v / np.sqrt(len(v))


def test_kt_feature_map():
    trials = 10
    n_features = [3, 10, 20]
    for n_f in n_features:
        for _ in range(trials):
            p = np.random.permutation(n_f)
        v = kt_feature_map(p, n_f)
        x = kt_inverse_feature_map(v, n_f)
        assert np.all(x == p), (x, p)


@njit
def kt_argmax(w, max_trials, n_features):
    p = np.zeros((max_trials, n_features), dtype=np.int64)
    scores = np.zeros(max_trials)
    for i in range(max_trials):
        p[i] = np.random.permutation(n_features)
        scores[i] = kt_feature_map(p[i], n_features).dot(w)
    best_trial = np.argmax(scores)
    return p[best_trial]


@njit
def kt_herding_permutations(n_samples, n_features):
    p = np.zeros((n_samples, n_features), dtype=np.int64)
    w = - np.ones(n_features * (n_features - 1) // 2)
    w /= np.sqrt(len(w))
    for i in range(n_samples):
        p[i] = kt_argmax(w, 10, n_features)
        w -= kt_feature_map(p[i], n_features)
    return p


def test_kt_convergence():
    n_samples_tests = [10, 100, 1000]
    n_features = 10
    error = []
    for n_samples in n_samples_tests:
        p = kt_herding_permutations(n_samples, n_features)
        feature_maps = np.array([kt_feature_map(p_i, n_features) for p_i in p])
        rmse = np.sqrt((feature_maps.mean(axis=0) ** 2).mean())
        error.append(rmse)
    assert np.all(np.diff(error) < 0)


def compute_bayesian_weights(p, kernel):
    n = p.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i <= j:
                k = kernel(p[i], p[j])
                K[i, j] = k
                K[j, i] = k
    z = np.full(n, 0.0)
    w = np.linalg.lstsq(K, z, rcond=None)[0]
    return w


def sbq_variance(K):
    z = np.full(K.shape[0], 0.5)
    w = np.linalg.lstsq(K, z, rcond=None)[0]
    return 0.5 - w.dot(z)


def update_kernel_matrix(K, p, new_p, kernel):
    K_new = K.copy()
    K_new[len(p), len(p)] = 1.0
    for i in range(len(p)):
        k = kernel(new_p, p[i])
        K_new[i, len(p)] = k
        K_new[len(p), i] = k
    return K_new


def sequential_bayesian_quadrature(n_points, n_features):
    num_trials = 10
    # start with a random permutation
    p = [np.random.permutation(n_features)]
    K = np.zeros((n_points, n_points))
    K[0, 0] = 1.0
    for i in range(1, n_points):
        trial_points = []
        trial_points_var = []
        for j in range(num_trials):
            # Create a trial kernel matrix
            trial_perm = np.random.permutation(n_features)
            trial_K = update_kernel_matrix(K, p, trial_perm, kt_kernel)
            trial_points_var.append(sbq_variance(trial_K[:i + 1, :i + 1]))
            trial_points.append((trial_perm, trial_K))
        best = np.argmin(trial_points_var)
        p.append(trial_points[best][0])
        K = trial_points[best][1]

    z = np.full(n_points, 0.5)
    w = np.linalg.lstsq(K, z, rcond=None)[0]
    return np.array(p), w

# print(sequential_bayesian_quadrature(20, 5))
