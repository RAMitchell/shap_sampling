import pytest
import datasets
import algorithms
import numpy as np
import sklearn


@pytest.mark.parametrize("data", [datasets.get_cal_housing(5, 10), datasets.get_regression(5, 10)])
@pytest.mark.parametrize("alg", [algorithms.castro, algorithms.castro_complement])
def test_efficiency(data, alg):
    model, X_background, X_foreground, exact_shap_values = data
    n_features = X_background.shape[1]
    shap_values = alg(X_background, X_foreground, model.predict,
                      algorithms.min_sample_size(alg, n_features))
    expected_value = model.predict(X_background).mean()
    y = model.predict(X_foreground)
    shap_sum = shap_values.sum(axis=1) + expected_value
    assert np.allclose(y, shap_sum)


@pytest.mark.parametrize("alg", [algorithms.castro, algorithms.castro_complement, algorithms.owen,
                                 algorithms.owen_complement])
def test_permutation(alg):
    X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
    # Train arbitrary model to get some coefficients
    mod = sklearn.linear_model.LinearRegression().fit(X, y)
    # Single background and foreground instance
    # Gives zero effect to features when they are 'off'
    # and the effect of the regression coefficient when they are 'on'
    X_background = np.zeros((1, X.shape[1]))
    X_foreground = np.ones((1, X.shape[1]))
    shap_values = alg(X_background, X_foreground, mod.predict,
                      algorithms.min_sample_size(alg, X.shape[1]) * 5)

    assert np.allclose(mod.coef_, shap_values, rtol=1e-04, atol=1e-04)


rows_predict_count = 0


# Test if any of our algorithms are cheating and using more function calls then they should
@pytest.mark.parametrize("alg", [algorithms.castro, algorithms.castro_complement, algorithms.owen,
                                 algorithms.owen_complement])
def test_num_function_calls(alg):
    global rows_predict_count

    def predict(X):
        global rows_predict_count
        rows_predict_count += X.shape[0]
        return np.zeros(X.shape[0])

    n_features = 5
    X_background = np.zeros((10, 5))
    X_foreground = np.zeros((2, 5))

    min_samples = algorithms.min_sample_size(alg, n_features)
    alg(X_background, X_foreground, predict, min_samples)
    assert rows_predict_count == 10 * 2 * 2 * min_samples
    rows_predict_count = 0


@pytest.mark.parametrize("alg", [algorithms.castro, algorithms.castro_complement, algorithms.owen,
                                 algorithms.owen_complement])
def test_basic(alg):
    def predict(x):
        return x[:, 0]

    n_features = 3

    X_background = np.zeros((2, n_features))
    X_foreground = np.ones((1, n_features))

    shap_values = alg(X_background, X_foreground, predict,
                      algorithms.min_sample_size(alg, n_features))
    assert shap_values[0][0] == 1.0
    assert shap_values[0][1] == 0.0
    assert shap_values[0][2] == 0.0
