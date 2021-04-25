from itertools import combinations
import numpy as np
from numba import njit


@njit
def discordant_pairs(a, b):
    n = len(a)
    assert len(b) == n
    discordant = 0
    a_inv = np.argsort(a)
    b_inv = np.argsort(b)
    for i in range(n):
        for j in range(i):
            if (a_inv[i] < a_inv[j] and b_inv[i] > b_inv[j]) or (
                    a_inv[i] > a_inv[j] and b_inv[i] < b_inv[j]):
                discordant += 1
    return discordant


def test_discordant_pairs():
    d = 20
    a = np.random.permutation(d)
    b = np.flip(a)
    dis = discordant_pairs(a, b)
    norm_dis = dis / ((d * (d - 1)) / 2)
    assert norm_dis == 1.0
    dis = discordant_pairs(a, a)
    norm_dis = dis / (d * (d - 1) / 2)
    assert norm_dis == 0.0


class KTKernel:
    def __call__(self, a, b):
        n = len(a)
        return 1.0 - 4 * discordant_pairs(a, b) / (n * (n - 1))

    def expected_value(self, d):
        return 0.0


class MallowsKernel:
    def __init__(self, l=5):
        self.l = l

    def __call__(self, a, b):
        d = len(a)
        norm = (d * (d - 1)) / 2
        return np.exp(-self.l * discordant_pairs(a, b) / norm)

    def expected_value(self, d):
        mgf_expected = 1.0
        norm = (d * (d - 1)) / 2
        for j in range(1, d + 1):
            mgf_expected *= (1 - np.exp(j * (-self.l / norm))) / (j * (1 - np.exp(-self.l / norm)))
        return mgf_expected


class SpearmanKernel:
    def __call__(self, a, b):
        return a.dot(b)

    def expected_value(self, d):
        return (d * (d - 1) ** 2) / 4


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


def kernel_argmax(samples, max_trials, n_features, kernel):
    p = np.zeros((max_trials, n_features), dtype=np.int64)
    scores = np.full(max_trials, kernel.expected_value(n_features))
    n = samples.shape[0]
    for i in range(max_trials):
        p[i] = np.random.permutation(n_features)
        for s in samples:
            scores[i] -= 1 / (n + 1) * kernel(p[i], s)
    best_trial = np.argmax(scores)
    return p[best_trial]


def kernel_herding(n_samples, n_features, kernel, max_trials):
    p = np.zeros((n_samples, n_features), dtype=np.int64)
    for i in range(n_samples):
        p[i] = kernel_argmax(p[:i], max_trials, n_features, kernel)
    return p


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


def sequential_bayesian_quadrature(n_points, n_features, kernel, num_trials):
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
            trial_K = update_kernel_matrix(K, p, trial_perm, kernel)
            trial_points_var.append(sbq_variance(trial_K[:i + 1, :i + 1]))
            trial_points.append((trial_perm, trial_K))
        best = np.argmin(trial_points_var)
        p.append(trial_points[best][0])
        K = trial_points[best][1]

    z = np.full(n_points, kernel.expected_value(n_features))
    w = np.linalg.lstsq(K, z, rcond=None)[0]
    w /= w.sum()
    return np.array(p), w


def discrepancy(p, w, kernel):
    n = p.shape[0]
    d = p.shape[1]
    if w is None:
        w = np.full(n, 1 / n)
    disc = kernel.expected_value(d)
    for i in range(n):
        disc -= 2 * w[i] * kernel.expected_value(d)
    for i in range(n):
        for j in range(n):
            disc += w[i] * w[j] * kernel(p[i], p[j])
    return np.sqrt(disc)


def plot_mallows_lambda():
    import matplotlib.pyplot as plt
    plt.style.use("seaborn")
    plt.rc('font', family='serif')
    x = np.linspace(0, 1, 100)
    lambdas = [0.5, 5.0, 50]
    plt.figure(figsize=(4 * 1.3, 3 * 1.3))
    for l in lambdas:
        y = np.exp(- x * l)
        plt.plot(x, y, label="$\lambda={}$".format(l))
    plt.legend()
    plt.xlabel("$n_{dis}(\sigma,\sigma')/\\binom{d}{2}$")
    plt.ylabel("$K_M(\sigma,\sigma')$")
    plt.tight_layout()
    plt.savefig('figures/kernel/mallows_lambda.png')
    plt.show()
