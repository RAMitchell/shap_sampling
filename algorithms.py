import numpy as np
from numba import njit
import sobol_seq


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
def _accumulate_samples_castro(phi, predictions, j):
    for foreground_idx in range(predictions.shape[0]):
        for sample_idx in range(predictions.shape[1]):
            phi[foreground_idx, j[sample_idx]] += predictions[foreground_idx][
                                                      sample_idx] / predictions.shape[1]


def estimate_shap_given_permutations(X_background, X_foreground, predict_function, p):
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
        _accumulate_samples_castro(phi, predictions, j)
        pred_off = pred_on

    return phi


def castro(X_background, X_foreground, predict_function, n_samples):
    n_features = X_background.shape[1]

    assert n_samples % (n_features + 1) == 0
    # castro is allowed to take 2 * more samples than owen as it reuses predictions
    samples_per_feature = 2 * (n_samples // (n_features + 1))
    p = np.zeros((samples_per_feature, n_features), dtype=np.int64)
    for i in range(samples_per_feature):
        p[i] = np.random.permutation(n_features)
    return estimate_shap_given_permutations(X_background, X_foreground, predict_function, p)


def castro_complement(X_background, X_foreground, predict_function, n_samples):
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


def castro_qmc(X_background, X_foreground, predict_function, n_samples):
    n_features = X_background.shape[1]

    assert n_samples % (n_features + 1) == 0
    # castro is allowed to take 2 * more samples than owen as it reuses predictions
    samples_per_feature = 2 * (n_samples // (n_features + 1))
    p = sobol_permutations(samples_per_feature, n_features)
    return estimate_shap_given_permutations(X_background, X_foreground, predict_function, p)


# Transform samples from [0,1) into permutations using Fisher Yates encoding
def real_samples_to_permutations(X, n_features):
    n_samples = X.shape[0]
    p = np.tile(np.arange(0, n_features), (n_samples, 1))
    for i in range(n_samples):
        for j in range(n_features - 1, 0, -1):
            k = min(int(X[i][j - 1] * (j + 1)), j)
            tmp = p[i][j]
            p[i][j] = p[i][k]
            p[i][k] = tmp
    return p


def sobol_permutations(n_samples, n_features):
    sobol = sobol_seq.i4_sobol_generate(n_features - 1, n_samples)
    return real_samples_to_permutations(sobol, n_features)


def castro_complement_qmc(X_background, X_foreground, predict_function, n_samples):
    n_features = X_background.shape[1]

    assert n_samples % (2 * (n_features + 1)) == 0
    # castro is allowed to take 2 * more samples than owen as it reuses predictions
    samples_per_feature = 2 * (n_samples // (n_features + 1))
    p = np.zeros((samples_per_feature, n_features), dtype=np.int64)

    p[0:samples_per_feature // 2] = sobol_permutations(samples_per_feature // 2, n_features)

    # Reverse samples
    for i in range(samples_per_feature // 2):
        p[i + samples_per_feature // 2] = np.flip(p[i])
    return estimate_shap_given_permutations(X_background, X_foreground, predict_function, p)


def get_lhs_permutations(n_samples, n_features):
    d_permutations = np.array([np.random.permutation(n_samples) for _ in range(n_features - 1)])
    X = np.zeros((n_samples, n_features - 1))
    for i in range(n_samples):
        for j in range(n_features - 1):
            X[i, j] = (d_permutations[j][i] + np.random.random()) / n_samples

    return real_samples_to_permutations(X, n_features)


def castro_lhs(X_background, X_foreground, predict_function, n_samples):
    n_features = X_background.shape[1]

    assert n_samples % (n_features + 1) == 0
    # castro is allowed to take 2 * more samples than owen as it reuses predictions
    samples_per_feature = 2 * (n_samples // (n_features + 1))
    p = get_lhs_permutations(samples_per_feature, n_features)
    return estimate_shap_given_permutations(X_background, X_foreground, predict_function, p)


@njit
def accumulate_samples_simple(y, n_features, phi, n_samples, n_foreground, s, mask):
    for foreground_idx in range(n_foreground):
        phi_on = np.zeros((n_features, n_features))
        phi_on_count = np.zeros((n_features, n_features))
        phi_off = np.zeros((n_features, n_features))
        phi_off_count = np.zeros((n_features, n_features))
        for sample_idx in range(n_samples):
            num_on = s[sample_idx]
            for j in range(n_features):
                if mask[sample_idx][j]:
                    phi_on[j, num_on - 1] += y[foreground_idx][sample_idx]
                    phi_on_count[j, num_on - 1] += 1
                else:
                    phi_off[j, num_on] += y[foreground_idx][sample_idx]
                    phi_off_count[j, num_on] += 1

        # avoid divide by 0
        for i in range(n_features):
            for j in range(n_features):
                if phi_on_count[i][j] == 0:
                    phi_on_count[i][j] = 1
                if phi_off_count[i][j] == 0:
                    phi_off_count[i][j] = 1

        strata_on = np.divide(phi_on, phi_on_count)
        strata_off = np.divide(phi_off, phi_off_count)

        mean_on = np.zeros(n_features)
        mean_off = np.zeros(n_features)
        for i in range(n_features):
            mean_on[i] = strata_on[i].mean()
            mean_off[i] = strata_off[i].mean()

        phi[foreground_idx] = mean_on - mean_off


def global_stratified(X_background, X_foreground, predict_function, n_samples):
    # Algorithm doesn't compute on/off pairs, so gets extra samples
    n_samples *= 2
    n_features = X_background.shape[1]
    phi = np.zeros((X_foreground.shape[0], n_features))
    assert n_samples % (n_features + 1) == 0

    # generate stratified samples uniformly
    mask = np.zeros((n_samples, n_features)).astype(bool)
    for i in range(n_samples):
        s_len = i % (n_features + 1)
        mask[i, 0:s_len] = True
        mask[i] = np.random.permutation(mask[i])

    masked_dataset = mask_dataset(mask, X_background, X_foreground)
    s = mask.sum(axis=1)
    y = predict_function(masked_dataset).reshape(
        (X_foreground.shape[0], mask.shape[0], X_background.shape[0]))
    # average the background samples
    y = y.mean(axis=2)
    accumulate_samples_simple(y, n_features, phi, n_samples, X_foreground.shape[0], s, mask)

    return phi


def global_stratified_complement(X_background, X_foreground, predict_function, n_samples):
    # Algorithm doesn't compute on/off pairs, so gets extra samples
    n_samples *= 2
    n_features = X_background.shape[1]
    phi = np.zeros((X_foreground.shape[0], n_features))
    assert n_samples % (2 * (n_features + 1)) == 0

    # generate stratified samples uniformly
    mask = np.zeros((n_samples, n_features)).astype(bool)
    for i in range(n_samples // 2):
        s_len = i % (n_features + 1)
        mask[i, 0:s_len] = True
        mask[i] = np.random.permutation(mask[i])

    # generate complement
    mask[n_samples // 2:] = np.invert(mask[0:n_samples // 2])

    masked_dataset = mask_dataset(mask, X_background, X_foreground)
    s = mask.sum(axis=1)
    y = predict_function(masked_dataset).reshape(
        (X_foreground.shape[0], mask.shape[0], X_background.shape[0]))
    # average the background samples
    y = y.mean(axis=2)
    accumulate_samples_simple(y, n_features, phi, n_samples, X_foreground.shape[0], s, mask)

    return phi


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


def min_sample_size(alg, n_features):
    if alg == castro:
        return n_features + 1
    elif alg == castro_qmc:
        return n_features + 1
    elif alg == castro_lhs:
        return n_features + 1
    elif alg == castro_complement:
        return 2 * (n_features + 1)
    elif alg == castro_complement_qmc:
        return 2 * (n_features + 1)
    elif alg == owen or alg == owen_complement:
        return n_features * 4
    elif alg == global_stratified:
        return n_features + 1
    elif alg == global_stratified_complement:
        return 2 * (n_features + 1)
    elif alg == castro_stratified:
        return 2 * (n_features ** 2)
    else:
        raise NotImplementedError()
