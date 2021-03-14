import pytest
import datasets
import algorithms
import numpy as np
import sklearn

all_algorithms = [algorithms.monte_carlo, algorithms.monte_carlo_antithetic, algorithms.owen,
                  algorithms.owen_complement, algorithms.castro_stratified,
                  algorithms.qmc_sobol, algorithms.qmc_sobol,
                  algorithms.kt_herding,
                  algorithms.castro_control_variate, algorithms.orthogonal, algorithms.fibonacci,
                  algorithms.monte_carlo_weighted, algorithms.sbq]

efficient_algorithms = [algorithms.monte_carlo, algorithms.monte_carlo_antithetic,
                        algorithms.qmc_sobol,
                        algorithms.kt_herding,
                        algorithms.orthogonal, algorithms.fibonacci,
                        algorithms.monte_carlo_weighted, algorithms.sbq]

algorithms_with_consistent_function_calls = [algorithms.monte_carlo,
                                             algorithms.monte_carlo_antithetic,
                                             algorithms.owen,
                                             algorithms.owen_complement,
                                             algorithms.castro_stratified, algorithms.qmc_sobol,
                                             algorithms.kt_herding,
                                             algorithms.orthogonal,
                                             algorithms.fibonacci,
                                             algorithms.monte_carlo_weighted, algorithms.sbq]

algorithms_exact_linear_model = [algorithms.monte_carlo, algorithms.monte_carlo_antithetic,
                                 algorithms.owen,
                                 algorithms.owen_complement, algorithms.castro_stratified,
                                 algorithms.qmc_sobol, algorithms.qmc_sobol,
                                 algorithms.kt_herding,
                                 algorithms.orthogonal, algorithms.fibonacci,
                                 algorithms.monte_carlo_weighted, algorithms.sbq]


@pytest.mark.parametrize("data", [datasets.get_cal_housing(5, 10), datasets.get_regression(5, 10)])
@pytest.mark.parametrize("alg",
                         efficient_algorithms)
def test_efficiency(data, alg):
    model, X_background, X_foreground, exact_shap_values = data
    n_features = X_background.shape[1]
    shap_values = alg(X_background, X_foreground, model.predict,
                      algorithms.min_sample_size(alg, n_features))
    expected_value = model.predict(X_background).mean()
    y = model.predict(X_foreground)
    shap_sum = shap_values.sum(axis=1) + expected_value
    assert np.allclose(y, shap_sum)


@pytest.mark.parametrize("alg", algorithms_exact_linear_model)
def test_linear_model(alg):
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
@pytest.mark.parametrize("alg", algorithms_with_consistent_function_calls)
def test_num_function_calls(alg):
    np.random.seed(978)
    global rows_predict_count

    def predict(X):
        global rows_predict_count
        rows_predict_count += X.shape[0]
        return np.random.random(X.shape[0])

    n_features = 5
    X_background = np.zeros((10, 5))
    X_foreground = np.zeros((2, 5))

    min_samples = algorithms.min_sample_size(alg, n_features)
    alg(X_background, X_foreground, predict, min_samples * 2)
    assert rows_predict_count == 10 * 2 * 2 * min_samples * 2
    rows_predict_count = 0


@pytest.mark.parametrize("alg", algorithms_exact_linear_model)
def test_basic(alg):
    def predict(x):
        return x[:, 0]

    n_features = 3

    X_background = np.zeros((2, n_features))
    X_foreground = np.ones((1, n_features))
    np.random.seed(11)
    shap_values = alg(X_background, X_foreground, predict,
                      algorithms.min_sample_size(alg, n_features))
    assert shap_values[0][0] == 1.0
    assert shap_values[0][1] == 0.0
    assert shap_values[0][2] == 0.0


def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())


def test_castro_stratified():
    import xgboost as xgb
    import shap
    np.random.seed(11)
    n_foreground = 5
    n_background = 10
    X, y = sklearn.datasets.make_regression(
        n_samples=10,
        n_features=3,
        noise=0.1,
        random_state=47)
    model = xgb.XGBRegressor().fit(X, y)
    X_foreground = X[:n_foreground]
    X_background = X[:n_background]
    tree_explainer = shap.TreeExplainer(model, X_background)
    exact_shap_values = tree_explainer.shap_values(X_foreground)

    errors = []
    for i in [1, 10, 100]:
        shap_values = algorithms.castro_stratified(X_background, X_foreground, model.predict,
                                                   algorithms.min_sample_size(
                                                       algorithms.castro_stratified,
                                                       3) * i)
        errors.append(rmse(exact_shap_values, shap_values))
    assert np.all(np.diff(errors) < 0)


