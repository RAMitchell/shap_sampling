import shap
import sklearn.datasets
import xgboost as xgb


def get_regression(n_foreground, n_background):
    X, y = sklearn.datasets.make_regression(
        n_samples=100,
        n_features=10,
        noise=0.1,
        random_state=47)
    model = xgb.XGBRegressor().fit(X, y)
    X_foreground = X[:n_foreground]
    X_background = X[:n_background]
    tree_explainer = shap.TreeExplainer(model, X_background)
    exact_shap_values = tree_explainer.shap_values(X_foreground)
    return (model, X_background, X_foreground, exact_shap_values)


def get_regression_small_features(n_foreground, n_background):
    X, y = sklearn.datasets.make_regression(
        n_samples=100,
        n_features=3,
        noise=0.1,
        random_state=47)
    model = xgb.XGBRegressor().fit(X, y)
    X_foreground = X[:n_foreground]
    X_background = X[:n_background]
    tree_explainer = shap.TreeExplainer(model, X_background)
    exact_shap_values = tree_explainer.shap_values(X_foreground)
    return (model, X_background, X_foreground, exact_shap_values)


def get_cal_housing(n_foreground, n_background):
    X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
    model = xgb.XGBRegressor().fit(X, y)
    X_foreground = X[:n_foreground]
    X_background = X[:n_background]
    tree_explainer = shap.TreeExplainer(model, X_background)
    exact_shap_values = tree_explainer.shap_values(X_foreground)
    return (model, X_background, X_foreground, exact_shap_values)
