import numpy as np
from numba import njit
import sobol_seq
import kernel_methods
from itertools import count
from math import cos, gamma, pi, sin, sqrt
from typing import Callable, Iterator, List


@njit
def mask_dataset(mask, X_background, X_foreground):
    n_samples = mask.shape[0]
    n_features = mask.shape[1]
    n_foreground = X_foreground.shape[0]
    n_background = X_background.shape[0]
    masked_dataset = np.zeros(
        (n_foreground, n_samples, n_background, n_features))
    for foreground_idx in range(n_foreground):
        for sample_idx in range(n_samples):
            for background_idx in range(n_background):
                masked_dataset[foreground_idx][sample_idx][background_idx] = X_background[
                    background_idx]
                masked_dataset[foreground_idx][sample_idx][background_idx][mask[sample_idx]] = \
                    X_foreground[foreground_idx][mask[sample_idx]]
    return masked_dataset.reshape((n_foreground * n_samples * n_background, n_features))


def owen(X_background, X_foreground, predict_function, n_samples):
    n_features = X_background.shape[1]
    phi = np.zeros((X_foreground.shape[0], n_features))

    runs = 2
    assert n_samples % (n_features * runs) == 0
    q_splits = (n_samples // (n_features * runs)) - 1
    s = []
    for _ in range(runs):
        for q_num in range(q_splits + 1):
            q = q_num / q_splits
            s.append(np.array(np.random.binomial(1, q, n_features)))
    mask = np.array(s, dtype=bool)
    for j in range(n_features):
        mask_j_on = mask.copy()
        mask_j_on[:, j] = 1
        masked_dataset_on = mask_dataset(mask_j_on, X_background, X_foreground)
        mask_j_off = mask_j_on
        mask_j_off[:, j] = 0
        masked_dataset_off = mask_dataset(mask_j_off, X_background, X_foreground)
        item = predict_function(masked_dataset_on) - predict_function(masked_dataset_off)
        item = item.reshape((X_foreground.shape[0], X_background.shape[0] * mask.shape[0]))
        phi[:, j] = np.mean(item, axis=1)

    return phi


def owen_complement(X_background, X_foreground, predict_function, n_samples):
    n_features = X_background.shape[1]
    phi = np.zeros((X_foreground.shape[0], n_features))

    runs = 2
    assert n_samples % (n_features * runs) == 0
    q_splits = (n_samples // (n_features * runs)) - 1
    s = []
    for _ in range(runs):
        for q_num in range(q_splits // 2 + 1):
            q = q_num / q_splits
            b = np.array(np.random.binomial(1, q, n_features))
            s.append(b)
            if q != 0.5:
                s.append(1 - b)

    mask = np.array(s, dtype=bool)
    for j in range(n_features):
        mask_j_on = mask.copy()
        mask_j_on[:, j] = 1
        masked_dataset_on = mask_dataset(mask_j_on, X_background, X_foreground)
        mask_j_off = mask_j_on
        mask_j_off[:, j] = 0
        masked_dataset_off = mask_dataset(mask_j_off, X_background, X_foreground)
        item = predict_function(masked_dataset_on) - predict_function(masked_dataset_off)
        item = item.reshape((X_foreground.shape[0], X_background.shape[0] * mask.shape[0]))
        phi[:, j] = np.mean(item, axis=1)

    return phi


@njit
def _accumulate_samples_castro(phi, predictions, j, weights=None):
    if weights == None:
        weights = np.full(predictions.shape[1], 1 / predictions.shape[1])
    for foreground_idx in range(predictions.shape[0]):
        for sample_idx in range(predictions.shape[1]):
            phi[foreground_idx, j[sample_idx]] += predictions[foreground_idx][
                                                      sample_idx] * weights[sample_idx]


def estimate_shap_given_permutations(X_background, X_foreground, predict_function, p, weights=None):
    n_features = X_background.shape[1]
    phi = np.zeros((X_foreground.shape[0], n_features))
    n_permutations = p.shape[0]
    mask = np.zeros((n_permutations, n_features), dtype=bool)
    masked_dataset = mask_dataset(mask, X_background, X_foreground)
    pred_off = predict_function(masked_dataset)
    for j in p.T:
        mask[range(n_permutations), j] = True
        masked_dataset = mask_dataset(mask, X_background, X_foreground)
        pred_on = predict_function(masked_dataset)
        predictions = (pred_on - pred_off).reshape(
            (X_foreground.shape[0], mask.shape[0], X_background.shape[0]))
        predictions = np.mean(predictions, axis=2)
        _accumulate_samples_castro(phi, predictions, j, weights)
        pred_off = pred_on

    return phi


def monte_carlo(X_background, X_foreground, predict_function, n_samples):
    n_features = X_background.shape[1]

    assert n_samples % (n_features + 1) == 0
    # castro is allowed to take 2 * more samples than owen as it reuses predictions
    samples_per_feature = 2 * (n_samples // (n_features + 1))
    p = np.zeros((samples_per_feature, n_features), dtype=np.int64)
    for i in range(samples_per_feature):
        p[i] = np.random.permutation(n_features)
    return estimate_shap_given_permutations(X_background, X_foreground, predict_function, p)


def monte_carlo_weighted(X_background, X_foreground, predict_function, n_samples):
    n_features = X_background.shape[1]
    assert n_samples % (n_features + 1) == 0
    # castro is allowed to take 2 * more samples than owen as it reuses predictions
    samples_per_feature = 2 * (n_samples // (n_features + 1))
    p = np.zeros((samples_per_feature, n_features), dtype=np.int64)
    for i in range(samples_per_feature):
        p[i] = np.random.permutation(n_features)
    weights = kernel_methods.compute_bayesian_weights(p, kernel_methods.kt_kernel)
    return estimate_shap_given_permutations(X_background, X_foreground, predict_function, p,
                                            weights)


def sbq(X_background, X_foreground, predict_function, n_samples):
    n_features = X_background.shape[1]
    assert n_samples % (n_features + 1) == 0
    samples_per_feature = 2 * (n_samples // (n_features + 1))
    p, w = kernel_methods.sequential_bayesian_quadrature(samples_per_feature, n_features)
    return estimate_shap_given_permutations(X_background, X_foreground, predict_function, p,
                                            w)


def monte_carlo_antithetic(X_background, X_foreground, predict_function, n_samples):
    n_features = X_background.shape[1]

    assert n_samples % (2 * (n_features + 1)) == 0
    # castro is allowed to take 2 * more samples than owen as it reuses predictions
    samples_per_feature = 2 * (n_samples // (n_features + 1))
    p = np.zeros((samples_per_feature, n_features), dtype=np.int64)

    # Forward samples
    for i in range(samples_per_feature // 2):
        p[i] = np.random.permutation(n_features)

    # Reverse samples
    for i in range(samples_per_feature // 2):
        p[i + samples_per_feature // 2] = np.flip(p[i])

    return estimate_shap_given_permutations(X_background, X_foreground, predict_function, p)


def qmc_sobol(X_background, X_foreground, predict_function, n_samples):
    n_features = X_background.shape[1]

    assert n_samples % (n_features + 1) == 0
    # castro is allowed to take 2 * more samples than owen as it reuses predictions
    samples_per_feature = 2 * (n_samples // (n_features + 1))
    p = sobol_sphere_permutations(samples_per_feature, n_features)
    return estimate_shap_given_permutations(X_background, X_foreground, predict_function, p)


def sobol_sphere_permutations(n_samples, n_features):
    sobol = sobol_seq.i4_sobol_generate(n_features, n_samples)

    return np.argsort(sobol, axis=1)


# sample with l ones and i off
def draw_castro_stratified_samples(n_samples, n_features, i, l):
    mask = np.zeros((n_samples, n_features - 1), dtype=bool)
    mask[:, 0:l] = True
    for sample_idx in range(n_samples):
        mask[sample_idx] = np.random.permutation(mask[sample_idx])
    mask = np.insert(mask, i, False, axis=1)
    return mask


def allocate_samples(variance, n_samples):
    variance_flat = variance.flatten()
    samples_out = np.zeros_like(variance_flat, dtype=int)
    sum_variance = variance_flat.sum()
    for i in range(len(variance_flat)):
        if (sum_variance > 0.0):
            samples_out[i] = np.round(variance_flat[i] / sum_variance * n_samples)
            sum_variance -= variance_flat[i]
            n_samples -= samples_out[i]
    return samples_out.reshape(variance.shape)


def castro_stratified(X_background, X_foreground, predict_function, n_samples):
    n_features = X_background.shape[1]
    phi = np.zeros((X_foreground.shape[0], n_features))

    assert n_samples % (2 * n_features ** 2) == 0
    m_i_l_exp = n_samples // (2 * n_features ** 2)
    for foreground_idx in range(X_foreground.shape[0]):

        stratified_phi = np.zeros((n_features, n_features))
        stratified_phi_count = np.zeros((n_features, n_features))
        stratified_variance = np.zeros((n_features, n_features))
        # Uniformly sample strata
        for i in range(n_features):
            for l in range(0, n_features):
                mask_off = draw_castro_stratified_samples(m_i_l_exp, n_features, i, l)
                masked_dataset_off = mask_dataset(mask_off, X_background, X_foreground[
                                                                          foreground_idx:foreground_idx + 1])
                pred_off = predict_function(masked_dataset_off)
                mask_on = mask_off
                mask_on[:, i] = True
                masked_dataset_on = mask_dataset(mask_on, X_background, X_foreground[
                                                                        foreground_idx:foreground_idx + 1])
                pred_on = predict_function(masked_dataset_on)
                # Average background samples
                y = (pred_on - pred_off).reshape(m_i_l_exp, X_background.shape[0]).mean(axis=1)
                stratified_phi[i, l] += y.sum()
                stratified_phi_count[i, l] += m_i_l_exp
                stratified_variance[i, l] = np.var(y, ddof=1)

        stratified_proportional_samples = allocate_samples(stratified_variance, n_samples // 2)
        # Sample highest variance regions
        for i in range(n_features):
            for l in range(0, n_features):
                m_i_l_st = stratified_proportional_samples[i][l]
                if m_i_l_st == 0:
                    continue
                mask_off = draw_castro_stratified_samples(m_i_l_st, n_features, i, l)
                masked_dataset_off = mask_dataset(mask_off, X_background, X_foreground[
                                                                          foreground_idx:foreground_idx + 1])
                pred_off = predict_function(masked_dataset_off)
                mask_on = mask_off
                mask_on[:, i] = True
                masked_dataset_on = mask_dataset(mask_on, X_background, X_foreground[
                                                                        foreground_idx:foreground_idx + 1])
                pred_on = predict_function(masked_dataset_on)
                # Average background samples
                y = (pred_on - pred_off).reshape(m_i_l_st, X_background.shape[0]).mean(axis=1)
                stratified_phi[i, l] += y.sum()
                stratified_phi_count[i, l] += m_i_l_st

        stratified_phi /= stratified_phi_count
        phi[foreground_idx] = stratified_phi.mean(axis=1).T

    return phi


def correlation(a, b):
    a_mean = a.mean()
    b_mean = b.mean()
    cov = 0.0
    var = 0.0
    for i in range(len(a)):
        cov += (a[i] - a_mean) * (b[i] - b_mean)
        var += (b[i] - b_mean) * (b[i] - b_mean)
    return cov / var


def castro_control_variate(X_background, X_foreground, predict_function, n_samples):
    n_features = X_background.shape[1]

    # Train tree model
    from sklearn import tree
    import shap
    from sklearn.utils import resample
    tree_training_X = resample(np.vstack((X_background, X_foreground)), n_samples=10000)
    for i in range(n_features):
        tree_training_X[:, i] = np.random.permutation(tree_training_X[:, i])
    tree_training_y = predict_function(tree_training_X)
    tree_model = tree.DecisionTreeRegressor().fit(tree_training_X, tree_training_y)
    # estimate correlation using predictions on background set
    beta = correlation(tree_model.predict(X_background), predict_function(X_background))
    difference_predict = lambda X: predict_function(X) - beta * tree_model.predict(X)
    tree_explainer = shap.TreeExplainer(tree_model, X_background)
    tree_phi = tree_explainer.shap_values(X_foreground, check_additivity=False)
    phi = monte_carlo(X_background, X_foreground, difference_predict, n_samples)
    return phi + tree_phi * beta


def kt_herding(X_background, X_foreground, predict_function, n_samples):
    n_features = X_background.shape[1]
    assert n_samples % (n_features + 1) == 0
    # castro is allowed to take 2 * more samples than owen as it reuses predictions
    samples_per_feature = 2 * (n_samples // (n_features + 1))
    p = kernel_methods.kt_herding_permutations(samples_per_feature, n_features)
    return estimate_shap_given_permutations(X_background, X_foreground, predict_function, p)


def _sample_sphere(ndim):
    vec = np.random.randn(ndim)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def get_orthogonal_vectors(n_orthogonal_samples, n_features):
    A = np.zeros((n_orthogonal_samples, n_features))
    n = np.ones(n_features)
    n /= np.linalg.norm(n)
    for i in range(n_orthogonal_samples):
        A[i] = _sample_sphere(n_features)
        A[i, :] -= np.dot(n, A[i, :]) * n
        for k in range(i):
            A[i, :] -= np.dot(A[k, :], A[i, :]) * A[k, :]
        A[i, :] = A[i, :] / np.linalg.norm(A[i, :])
    return A


def _gram_schmidt_permutations(n_orthogonal_samples, n_features):
    assert n_orthogonal_samples < n_features
    A = get_orthogonal_vectors(n_orthogonal_samples, n_features)
    p = np.zeros(A.shape, dtype=np.int64)
    for i in range(n_orthogonal_samples):
        p[i] = np.argsort(A[i])

    return p


def _orthogonal_permutations(n_samples, n_features):
    p = np.zeros((n_samples, n_features), dtype=np.int64)
    k = 8
    i = 0
    while i < n_samples // 2:
        n_orthogonal_samples = min(min(k, n_features - 1), n_samples // 2 - i)
        p[i:i + n_orthogonal_samples] = _gram_schmidt_permutations(n_orthogonal_samples, n_features
                                                                   )
        i += n_orthogonal_samples

    # Reverse samples
    for i in range(n_samples // 2):
        p[i + n_samples // 2] = np.flip(p[i])
    return p


def orthogonal(X_background, X_foreground, predict_function, n_samples):
    n_features = X_background.shape[1]
    assert n_samples % (2 * (n_features + 1)) == 0
    # castro is allowed to take 2 * more samples than owen as it reuses predictions
    samples_per_feature = 2 * (n_samples // (n_features + 1))
    p = _orthogonal_permutations(samples_per_feature, n_features)
    return estimate_shap_given_permutations(X_background, X_foreground, predict_function, p)


def orthogonal_weighted(X_background, X_foreground, predict_function, n_samples):
    n_features = X_background.shape[1]
    assert n_samples % (2 * (n_features + 1)) == 0
    # castro is allowed to take 2 * more samples than owen as it reuses predictions
    samples_per_feature = 2 * (n_samples // (n_features + 1))
    p = _orthogonal_permutations(samples_per_feature, n_features)
    w = kernel_methods.compute_bayesian_weights(p, kernel_methods.kt_kernel)
    return estimate_shap_given_permutations(X_background, X_foreground, predict_function, p, w)


def _int_sin_m(x: float, m: int) -> float:
    """Computes the integral of sin^m(t) dt from 0 to x recursively"""
    if m == 0:
        return x
    elif m == 1:
        return 1 - cos(x)
    else:
        return (m - 1) / m * _int_sin_m(x, m - 2) - cos(x) * sin(x) ** (
                m - 1
        ) / m


def _primes() -> Iterator[int]:
    """Returns an infinite generator of prime numbers"""
    yield from (2, 3, 5, 7)
    composites = {}
    ps = _primes()
    next(ps)
    p = next(ps)
    assert p == 3
    psq = p * p
    for i in count(9, 2):
        if i in composites:  # composite
            step = composites.pop(i)
        elif i < psq:  # prime
            yield i
            continue
        else:  # composite, = p*p
            assert i == psq
            step = 2 * p
            p = next(ps)
            psq = p * p
        i += step
        while i in composites:
            i += step
        composites[i] = step


def _inverse_increasing(
        func: Callable[[float], float],
        target: float,
        lower: float,
        upper: float,
        atol: float = 1e-10,
) -> float:
    """Returns func inverse of target between lower and upper

    inverse is accurate to an absolute tolerance of atol, and
    must be monotonically increasing over the interval lower
    to upper
    """
    mid = (lower + upper) / 2
    approx = func(mid)
    while abs(approx - target) > atol:
        if approx > target:
            upper = mid
        else:
            lower = mid
        mid = (upper + lower) / 2
        approx = func(mid)
    return mid


def uniform_hypersphere(d: int, n: int) -> List[List[float]]:
    """Generate n points over the d dimensional hypersphere"""
    assert d > 1
    assert n > 0
    points = [[1 for _ in range(d)] for _ in range(n)]
    for i in range(n):
        t = 2 * pi * i / n
        points[i][0] *= sin(t)
        points[i][1] *= cos(t)
    for dim, prime in zip(range(2, d), _primes()):
        offset = sqrt(prime)
        mult = gamma(dim / 2 + 0.5) / gamma(dim / 2) / sqrt(pi)

        def dim_func(y):
            return mult * _int_sin_m(y, dim - 1)

        for i in range(n):
            deg = _inverse_increasing(dim_func, i * offset % 1, 0, pi)
            for j in range(dim):
                points[i][j] *= sin(deg)
            points[i][dim] *= cos(deg)
    return np.array(points)


def zero_sum_projection(d):
    basis = np.array([[1.0] * i + [-i] + [0.0] * (d - i - 1) for i in range(1, d)])
    return np.array([v / np.linalg.norm(v) for v in basis])


def fibonacci(X_background, X_foreground, predict_function, n_samples):
    n_features = X_background.shape[1]
    assert n_samples % (n_features + 1) == 0
    samples_per_feature = 2 * (n_samples // (n_features + 1))
    sphere_points = uniform_hypersphere(n_features - 1, samples_per_feature)
    basis = zero_sum_projection(n_features)
    projected_sphere_points = sphere_points.dot(basis)
    p = np.zeros((samples_per_feature, n_features), dtype=np.int64)
    for i in range(samples_per_feature):
        p[i] = np.argsort(projected_sphere_points[i])
    return estimate_shap_given_permutations(X_background, X_foreground, predict_function, p)


def min_sample_size(alg, n_features):
    if alg == monte_carlo:
        return n_features + 1
    elif alg == monte_carlo_weighted:
        return n_features + 1
    elif alg == sbq:
        return n_features + 1
    elif alg == qmc_sobol:
        return n_features + 1
    elif alg == fibonacci:
        return n_features + 1
    elif alg == kt_herding:
        return n_features + 1
    elif alg == castro_control_variate:
        return n_features + 1
    elif alg == monte_carlo_antithetic:
        return 2 * (n_features + 1)
    elif alg == orthogonal:
        return 2 * (n_features + 1)
    elif alg == orthogonal_weighted:
        return 2 * (n_features + 1)
    elif alg == owen or alg == owen_complement:
        return n_features * 4
    elif alg == castro_stratified:
        return 2 * (n_features ** 2)
    else:
        raise NotImplementedError()
