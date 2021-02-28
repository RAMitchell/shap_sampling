import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import algorithms
import datasets
import pandas as pd
import seaborn as sns

matplotlib.use('agg')
plt.style.use("seaborn")
plt.rc('font', family='serif')


def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())


def plot_experiments():
    repeats = 25
    foreground_examples = 10
    background_examples = 100
    max_evals = 100000
    datasets_set = {
        "make_regression": datasets.get_regression(foreground_examples, background_examples),
        "cal_housing": datasets.get_cal_housing(foreground_examples, background_examples),
        "adult": datasets.get_adult(foreground_examples, background_examples),
        "breast_cancer": datasets.get_breast_cancer(foreground_examples, background_examples),
    }
    algorithms_set = {
        "Castro": algorithms.monte_carlo,
        "Castro-Complement": algorithms.monte_carlo_antithetic,
        # "Castro-LHS": algorithms.castro_lhs,
    }
    algorithms_set = {
        # "Castro": algorithms.castro,
        "Castro-Orthogonal": algorithms.orthogonal,
        "Castro-Complement": algorithms.monte_carlo_antithetic,
        "Fibonacci": algorithms.fibonacci,
        # "Castro-ControlVariate": algorithms.castro_control_variate,
        # "Castro-QMC": algorithms.castro_qmc,
        # "KT-Herding": algorithms.kt_herding,
        # "Spearman-Herding": algorithms.spearman_herding,
        # "Spearman-Herding-Exact": algorithms.spearman_herding_exact,
    }

    deterministic_algorithms = ["Castro-QMC","Fibonacci"]

    seed = 32
    np.random.seed(seed)
    cp.random.seed(seed)
    for data_name, data in datasets_set.items():
        model, X_background, X_foreground, exact_shap_values = data
        model_predict = lambda X: model.get_booster().inplace_predict(X, predict_type='margin')
        n_features = X_background.shape[1]
        df = pd.DataFrame(columns=["Algorithm", "Function evals", "Trial", "rmse"])
        for alg_name, alg in algorithms_set.items():
            min_samples = algorithms.min_sample_size(alg, n_features)
            eval_schedule = [10 ** (x / 5) for x in range(1, 20)]
            eval_schedule = (np.round(np.divide(eval_schedule, min_samples)) * min_samples).astype(
                int)
            eval_schedule = eval_schedule[eval_schedule >= min_samples]
            eval_schedule = eval_schedule[eval_schedule <= max_evals]

            for evals in tqdm(eval_schedule, desc="Dataset - " + data_name + ", Alg - " + alg_name):
                required_repeats = repeats
                if alg_name in deterministic_algorithms:
                    required_repeats = 1
                for i in range(required_repeats):
                    shap_values = alg(X_background, X_foreground,
                                      model_predict,
                                      evals)
                    df = df.append({"Algorithm": alg_name, "n_permutations": evals/(n_features+1), "Trial": i,
                                    "rmse": rmse(shap_values, exact_shap_values)},
                                   ignore_index=True)
        sns.lineplot(data=df, x="n_permutations", y="rmse", hue="Algorithm")
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('figures/' + data_name + '_shap.png')
        plt.clf()


import cProfile, pstats, io
from pstats import SortKey

pr = cProfile.Profile()
pr.enable()
plot_experiments()
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(20)
print(s.getvalue())
