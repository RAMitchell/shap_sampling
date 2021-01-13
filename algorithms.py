import numpy as np
from math import lgamma
import cupy as cp
from numba import njit


# FIXME: foreground indices
# def mask_dataset(mask, X_background, X_foreground):
#     n_samples = mask.shape[0]
#     n_foreground = X_foreground.shape[0]
#     n_background = X_background.shape[0]
#     n_features = mask.shape[1]
#     masked_dataset = cp.zeros(
#         (n_foreground* n_samples* n_background, n_features))
#
#     mask_kernel = """
#     size_t col = i % ncols;
#     size_t background_row =(i/ncols) % nbackground_rows;
#     size_t sample_idx = i/(ncols*nbackground_rows);
#     out = mask[sample_idx*ncols + col] ? foreground[col] : background[background_row*ncols + col];
#     """
#     mask_foreground = cp.ElementwiseKernel(
#         'raw X background, raw Y foreground, raw Z mask, int64 ncols, int64 nbackground_rows',
#         'W out', mask_kernel, 'mask_foreground')
#     mask_foreground(cp.array(X_background), cp.array(X_foreground), cp.array(mask), n_features,
#     n_background, masked_dataset)
#     return masked_dataset.get()

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
def accumulate_samples_castro(phi, predictions, j):
    for foreground_idx in range(predictions.shape[0]):
        for sample_idx in range(predictions.shape[1]):
            phi[foreground_idx, j[sample_idx]] += predictions[foreground_idx][
                                                      sample_idx] / predictions.shape[1]


def castro(X_background, X_foreground, predict_function, n_samples):
    n_features = X_background.shape[1]
    phi = np.zeros((X_foreground.shape[0], n_features))

    assert n_samples % (n_features + 1) == 0
    # castro is allowed to take 2 * more samples than owen as it reuses predictions
    samples_per_feature = 2 * (n_samples // (n_features + 1))
    p = np.zeros((samples_per_feature, n_features), dtype=np.int64)
    for i in range(samples_per_feature):
        p[i] = np.random.permutation(n_features)
    mask = np.zeros((samples_per_feature, n_features), dtype=bool)
    masked_dataset = mask_dataset(mask, X_background, X_foreground)
    pred_off = predict_function(masked_dataset)
    for j in p.T:
        mask[range(samples_per_feature), j] = True
        masked_dataset = mask_dataset(mask, X_background, X_foreground)
        pred_on = predict_function(masked_dataset)
        predictions = (pred_on - pred_off).reshape(
            (X_foreground.shape[0], mask.shape[0], X_background.shape[0]))
        predictions = np.mean(predictions, axis=2)
        accumulate_samples_castro(phi, predictions, j)
        pred_off = pred_on

    return phi


def castro_complement(X_background, X_foreground, predict_function, n_samples):
    n_features = X_background.shape[1]
    phi = np.zeros((X_foreground.shape[0], n_features))

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

    mask = np.zeros((samples_per_feature, n_features), dtype=bool)
    masked_dataset = mask_dataset(mask, X_background, X_foreground)
    pred_off = predict_function(masked_dataset)
    for j in p.T:
        mask[range(samples_per_feature), j] = True
        masked_dataset = mask_dataset(mask, X_background, X_foreground)
        pred_on = predict_function(masked_dataset)
        predictions = (pred_on - pred_off).reshape(
            (X_foreground.shape[0], mask.shape[0], X_background.shape[0]))
        predictions = np.mean(predictions, axis=2)
        accumulate_samples_castro(phi, predictions, j)
        pred_off = pred_on

    return phi


@njit
def W(s, n):
    return np.exp(n * np.log(2.0) + lgamma(s + 1) - lgamma(n + 1) + lgamma(n - s))


@njit
def accumulate_samples_simple(y, n_features, phi, n_samples, n_foreground, s, mask):
    for foreground_idx in range(n_foreground):
        for sample_idx in range(n_samples):
            num_on = s[sample_idx]
            for j in range(n_features):
                if mask[sample_idx][j]:
                    phi[foreground_idx][j] += y[foreground_idx][sample_idx] * W(num_on - 1,
                                                                                n_features) / \
                                              n_samples
                else:
                    phi[foreground_idx][j] -= y[foreground_idx][sample_idx] * W(num_on,
                                                                                n_features) / \
                                              n_samples


def simple(X_background, X_foreground, predict_function, n_samples):
    n_features = X_background.shape[1]
    phi = np.zeros((X_foreground.shape[0], n_features))

    mask = np.random.binomial(1, 0.5, (n_samples, n_features))
    masked_dataset = mask_dataset(mask, X_background, X_foreground)
    s = mask.sum(axis=1)
    y = predict_function(masked_dataset).reshape(
        (X_foreground.shape[0], mask.shape[0], X_background.shape[0]))
    # average the background samples
    y = y.mean(axis=2)
    accumulate_samples_simple(y, n_features, phi, n_samples, X_foreground.shape[0], s, mask)

    return phi


def min_sample_size(alg, n_features):
    if alg == castro:
        return n_features + 1
    elif alg == castro_complement:
        return 2 * (n_features + 1)
    elif alg == owen or alg == owen_complement:
        return n_features * 4
    elif alg == simple:
        return 10
    else:
        raise NotImplementedError()
