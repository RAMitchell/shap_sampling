import numpy as np
from numba import njit
import kernel_methods
import sobol_sphere


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


# base class
class ShapleyEstimator:
    def max_evals(self, num_features):
        return np.iinfo(np.int64).max


class Owen(ShapleyEstimator):
    def __init__(self, runs=2):
        self.runs = runs

    def shap_values(self, X_background, X_foreground, predict_function, n_samples):
        n_features = X_background.shape[1]
        phi = np.zeros((X_foreground.shape[0], n_features))

        assert n_samples % self.min_samples(n_features) == 0
        q_splits = (n_samples // (n_features * self.runs)) - 1
        s = []
        for _ in range(self.runs):
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

    def min_samples(self, n_features):
        return 2 * (self.runs * n_features)


class OwenHalved(ShapleyEstimator):
    def __init__(self, runs=2):
        self.runs = runs

    def shap_values(self, X_background, X_foreground, predict_function, n_samples):
        n_features = X_background.shape[1]
        phi = np.zeros((X_foreground.shape[0], n_features))

        assert n_samples % self.min_samples(n_features) == 0
        q_splits = (n_samples // (n_features * self.runs)) - 1
        s = []
        for _ in range(self.runs):
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

    def min_samples(self, n_features):
        return 2 * (self.runs * n_features)


@njit
def _accumulate_samples(phi, predictions, j, weights=None):
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
        _accumulate_samples(phi, predictions, j, weights)
        pred_off = pred_on

    return phi


class MonteCarlo(ShapleyEstimator):
    def shap_values(self, X_background, X_foreground, predict_function, n_samples):
        n_features = X_background.shape[1]
        assert n_samples % self.min_samples(n_features) == 0
        # allowed to take 2 * samples as it reuses predictions
        samples_per_feature = 2 * (n_samples // self.min_samples(n_features))
        p = np.zeros((samples_per_feature, n_features), dtype=np.int64)
        for i in range(samples_per_feature):
            p[i] = np.random.permutation(n_features)
        return estimate_shap_given_permutations(X_background, X_foreground, predict_function, p)

    def min_samples(self, n_features):
        return n_features + 1


def get_antithetic_permutations(n, d):
    p = np.zeros((n, d), dtype=np.int64)
    # Forward samples
    for i in range(n // 2):
        p[i] = np.random.permutation(d)

    # Reverse samples
    for i in range(n // 2):
        p[i + n // 2] = np.flip(p[i])
    return p


class MonteCarloAntithetic(ShapleyEstimator):
    def shap_values(self, X_background, X_foreground, predict_function, n_samples):
        n_features = X_background.shape[1]
        assert n_samples % self.min_samples(n_features) == 0
        # allowed to take 2 * samples as it reuses predictions
        samples_per_feature = 2 * (n_samples // (n_features + 1))
        p = get_antithetic_permutations(samples_per_feature, n_features)
        return estimate_shap_given_permutations(X_background, X_foreground, predict_function, p)

    def min_samples(self, n_features):
        return 2 * (n_features + 1)


class BayesianQuadrature(ShapleyEstimator):
    def __init__(self, kernel):
        self.kernel = kernel

    def shap_values(self, X_background, X_foreground, predict_function, n_samples):
        n_features = X_background.shape[1]
        samples_per_feature = 2 * (n_samples // (n_features + 1))
        p = np.zeros((samples_per_feature, n_features), dtype=np.int64)
        for i in range(samples_per_feature):
            p[i] = np.random.permutation(n_features)
        weights = kernel_methods.compute_bayesian_weights(p, self.kernel)
        return estimate_shap_given_permutations(X_background, X_foreground, predict_function, p,
                                                weights)

    def min_samples(self, n_features):
        return n_features + 1


class SequentialBayesianQuadrature(ShapleyEstimator):
    def __init__(self, kernel, num_trials=25):
        self.kernel = kernel
        self.num_trials = num_trials

    def shap_values(self, X_background, X_foreground, predict_function, n_samples):
        n_features = X_background.shape[1]
        assert n_samples % self.min_samples(n_features) == 0
        # allowed to take 2 * samples as it reuses predictions
        samples_per_feature = 2 * (n_samples // (n_features + 1))
        p, w = kernel_methods.sequential_bayesian_quadrature(samples_per_feature, n_features,
                                                             self.kernel, self.num_trials)
        return estimate_shap_given_permutations(X_background, X_foreground, predict_function, p,
                                                w)

    def min_samples(self, n_features):
        return n_features + 1

    def max_evals(self, n_features):
        return (n_features + 1) * 100


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


class Stratified(ShapleyEstimator):
    def shap_values(self, X_background, X_foreground, predict_function, n_samples):
        n_features = X_background.shape[1]
        phi = np.zeros((X_foreground.shape[0], n_features))

        assert n_samples % self.min_samples(n_features) == 0
        m_i_l_exp = n_samples // self.min_samples(n_features)
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

    def min_samples(self, n_features):
        return 2 * n_features ** 2


def correlation(a, b):
    a_mean = a.mean()
    b_mean = b.mean()
    cov = 0.0
    var = 0.0
    for i in range(len(a)):
        cov += (a[i] - a_mean) * (b[i] - b_mean)
        var += (b[i] - b_mean) * (b[i] - b_mean)
    return cov / var


class ControlVariate(ShapleyEstimator):
    def __init__(self, sampling_algorithm, training_samples):
        self.sampling_algorithm = sampling_algorithm
        self.training_samples=training_samples

    def shap_values(self, X_background, X_foreground, predict_function, n_samples):
        n_features = X_background.shape[1]

        # Train tree model
        from sklearn import tree
        import shap
        from sklearn.utils import resample
        tree_training_X = resample(np.vstack((X_background, X_foreground)), n_samples=self.training_samples)
        for i in range(n_features):
            tree_training_X[:, i] = np.random.permutation(tree_training_X[:, i])
        tree_training_y = predict_function(tree_training_X)
        tree_model = tree.DecisionTreeRegressor().fit(tree_training_X, tree_training_y)
        # estimate correlation using predictions on background set
        beta = correlation(tree_model.predict(X_background), predict_function(X_background))
        beta=0.05
        difference_predict = lambda X: predict_function(X) - beta * tree_model.predict(X)
        tree_explainer = shap.TreeExplainer(tree_model, X_background)
        tree_phi = tree_explainer.shap_values(X_foreground, check_additivity=False)
        phi = self.sampling_algorithm.shap_values(X_background, X_foreground, difference_predict,
                                                  n_samples)

        return phi + tree_phi * beta

    def min_samples(self, n_features):
        return self.sampling_algorithm.min_samples(n_features)


class KernelHerding(ShapleyEstimator):
    def __init__(self, kernel, max_trials=25):
        self.kernel = kernel
        self.max_trials = max_trials

    def shap_values(self, X_background, X_foreground, predict_function, n_samples):
        n_features = X_background.shape[1]
        assert n_samples % self.min_samples(n_features) == 0
        samples_per_feature = 2 * (n_samples // self.min_samples(n_features))
        p = kernel_methods.kernel_herding(samples_per_feature, n_features, self.kernel,
                                          self.max_trials)
        return estimate_shap_given_permutations(X_background, X_foreground, predict_function, p)

    def min_samples(self, n_features):
        return n_features + 1


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
    k = n_features - 1
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


class OrthogonalSphericalCodes(ShapleyEstimator):
    def shap_values(self, X_background, X_foreground, predict_function, n_samples):
        n_features = X_background.shape[1]
        assert n_samples % self.min_samples(n_features) == 0
        samples_per_feature = 2 * (n_samples // self.min_samples(n_features))
        p = _orthogonal_permutations(samples_per_feature, n_features)
        return estimate_shap_given_permutations(X_background, X_foreground, predict_function, p)

    def min_samples(self, n_features):
        return n_features + 1


class Sobol(ShapleyEstimator):
    def shap_values(self, X_background, X_foreground, predict_function, n_samples):
        n_features = X_background.shape[1]
        assert n_samples % self.min_samples(n_features) == 0
        samples_per_feature = 2 * (n_samples // self.min_samples(n_features))
        p = sobol_sphere.sobol_permutations(samples_per_feature, n_features)
        return estimate_shap_given_permutations(X_background, X_foreground, predict_function, p)

    def min_samples(self, n_features):
        return n_features + 1
