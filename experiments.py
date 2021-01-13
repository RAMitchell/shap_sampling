import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import algorithms
import datasets

matplotlib.use('agg')
plt.style.use("seaborn")
plt.rc('font', family='serif')


def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())


def plot_experiments():
    repeats = 10
    foreground_examples = 10
    background_examples = 10
    max_evals = 1000
    datasets_set = {
        "make_regression": datasets.get_regression(foreground_examples, background_examples),
        "cal_housing": datasets.get_cal_housing(foreground_examples, background_examples)}
    algorithms_set = {"Owen": algorithms.owen, "Owen-Complement": algorithms.owen_complement,
                      "Castro": algorithms.castro,
                      "Castro-Complement": algorithms.castro_complement}
    seed = 31
    np.random.seed(seed)
    cp.random.seed(seed)
    for data_name, data in datasets_set.items():
        model, X_background, X_foreground, exact_shap_values = data
        n_features = X_background.shape[1]
        for alg_name, alg in algorithms_set.items():
            errors = []
            min_samples = algorithms.min_sample_size(alg, n_features)
            eval_schedule = range(min_samples, max_evals, min_samples * 4)
            for evals in tqdm(eval_schedule, desc="Dataset - " + data_name + ", Alg - " + alg_name):
                repeats_error = []
                for _ in range(repeats):
                    shap_values = alg(X_background, X_foreground,
                                      model.get_booster().inplace_predict,
                                      evals)
                    repeats_error.append(rmse(shap_values, exact_shap_values))
                errors.append(np.mean(repeats_error))
            plt.plot(eval_schedule, errors, label=alg_name)
        plt.legend()
        plt.xlabel('function_evals')
        plt.ylabel('rmse')
        plt.tight_layout()
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
