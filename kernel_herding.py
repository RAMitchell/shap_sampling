from itertools import combinations
import numpy as np
from numba import njit


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
