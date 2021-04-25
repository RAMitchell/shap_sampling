import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import algorithms
import datasets
import pandas as pd
import seaborn as sns
import kernel_methods
import sobol_sphere
from concurrent import futures
from itertools import repeat
import joblib

mem = joblib.Memory(location='./tmp', verbose=1)

matplotlib.use('agg')
plt.style.use("seaborn")
plt.rc('font', family='serif')


def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())


def get_eval_schedule(min_samples, max_evals):
    eval_schedule = [10 ** (x / 5) for x in range(1, 20)]
    eval_schedule = (np.round(np.divide(eval_schedule, min_samples)) * min_samples).astype(
        int)
    eval_schedule = eval_schedule[eval_schedule >= min_samples]
    eval_schedule = eval_schedule[eval_schedule <= max_evals]
    # remove duplicates
    return list(dict.fromkeys(eval_schedule))


def get_partial_results(alg, alg_name, num_evals, required_repeats, data, data_name):
    df = pd.DataFrame(columns=["Dataset", "Algorithm", "Function evals", "Trial", "rmse"])
    model, X_background, X_foreground, exact_shap_values = data
    if num_evals > alg.max_evals(X_background.shape[1]):
        return df
    model_predict = lambda X: model.get_booster().inplace_predict(X, predict_type='margin')

    for trial_i in range(required_repeats):
        shap_values = alg.shap_values(X_background, X_foreground,
                                      model_predict,
                                      num_evals)

        df = df.append(
            {"Dataset": data_name, "Algorithm": alg_name, "marginal_evals": num_evals,
             "Trial": trial_i,
             "rmse": rmse(shap_values, exact_shap_values)},
            ignore_index=True)
    return df


@mem.cache
def run_experiments(datasets_set, algorithms_set, repeats,
                    max_evals):
    deterministic_algorithms = ["Fibonacci"]

    seed = 33
    np.random.seed(seed)
    cp.random.seed(seed)
    df = pd.DataFrame(columns=["Dataset", "Algorithm", "Function evals", "Trial", "rmse"])
    for data_name, data in datasets_set.items():
        model, X_background, X_foreground, exact_shap_values = data
        n_features = X_background.shape[1]
        for alg_name, alg in algorithms_set.items():
            eval_schedule = get_eval_schedule(alg.min_samples(n_features), max_evals)
            print("Dataset - " + data_name + ", Alg - " + alg_name)
            required_repeats = repeats
            if alg_name in deterministic_algorithms:
                required_repeats = 1
            with futures.ThreadPoolExecutor() as executor:
                for result in executor.map(get_partial_results, repeat(alg), repeat(alg_name),
                                           eval_schedule, repeat(required_repeats), repeat(data),
                                           repeat(data_name)):
                    df = df.append(result)
    return df


def plot_experiments(name, df):
    for d in df["Dataset"].unique():
        plt.figure(figsize=(4 * 1.3, 3 * 1.3))
        sns.lineplot(data=df.loc[df["Dataset"] == d], x="marginal_evals", y="rmse", hue="Algorithm")
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('figures/' + name + '_' + d + '_shap.png')
        plt.clf()


def kernel_experiments():
    repeats = 25
    foreground_examples = 10
    background_examples = 100
    max_evals = 5000
    datasets_set = {
        "make_regression": datasets.get_regression(foreground_examples, background_examples),
        "cal_housing": datasets.get_cal_housing(foreground_examples, background_examples),
        "adult": datasets.get_adult(foreground_examples, background_examples),
        "breast_cancer": datasets.get_breast_cancer(foreground_examples, background_examples),
    }
    algorithms_set = {
        "Mallows-Herding-0.5": algorithms.KernelHerding(kernel_methods.MallowsKernel(l=0.5)),
        "Mallows-Herding-5": algorithms.KernelHerding(kernel_methods.MallowsKernel(l=5)),
        "Mallows-Herding-50": algorithms.KernelHerding(kernel_methods.MallowsKernel(l=50)),
        "KT-Herding": algorithms.KernelHerding(kernel_methods.KTKernel()),
        "Spearman-Herding": algorithms.KernelHerding(kernel_methods.SpearmanKernel()),
    }

    df = run_experiments(datasets_set, algorithms_set, repeats, max_evals)
    plot_experiments("kernel/kernel", df)


def kernel_argmax_experiments():
    repeats = 25
    foreground_examples = 10
    background_examples = 100
    max_evals = 5000
    datasets_set = {
        "cal_housing": datasets.get_cal_housing(foreground_examples, background_examples),
    }
    algorithms_set = {
        "Mallows-5-trials": algorithms.KernelHerding(kernel_methods.MallowsKernel(),
                                                     max_trials=5),
        "Mallows-10-trials": algorithms.KernelHerding(kernel_methods.MallowsKernel(),
                                                      max_trials=10),
        "Mallows-25-trials": algorithms.KernelHerding(kernel_methods.MallowsKernel(),
                                                      max_trials=25),
        "Mallows-50-trials": algorithms.KernelHerding(kernel_methods.MallowsKernel(),
                                                      max_trials=50),
    }

    df = run_experiments(datasets_set, algorithms_set, repeats, max_evals)
    plot_experiments("kernel/kernel_trials", df)


def incumbent_experiments():
    repeats = 25
    foreground_examples = 10
    background_examples = 100
    max_evals = 5000
    datasets_set = {
        "make_regression": datasets.get_regression(foreground_examples, background_examples),
        "cal_housing": datasets.get_cal_housing(foreground_examples, background_examples),
        "adult": datasets.get_adult(foreground_examples, background_examples),
        "breast_cancer": datasets.get_breast_cancer(foreground_examples, background_examples),
    }
    algorithms_set = {
        "MC": algorithms.MonteCarlo(),
        "MC-antithetic": algorithms.MonteCarloAntithetic(),
        "Stratified": algorithms.Stratified(),
        "Owen": algorithms.Owen(),
        "Owen-Halved": algorithms.OwenHalved(),
    }

    df = run_experiments(datasets_set, algorithms_set, repeats, max_evals)
    plot_experiments("incumbent/incumbent", df)


def new_experiments():
    repeats = 25
    foreground_examples = 10
    background_examples = 100
    max_evals = 5000
    datasets_set = {
        "make_regression": datasets.get_regression(foreground_examples, background_examples),
        "cal_housing": datasets.get_cal_housing(foreground_examples, background_examples),
        "adult": datasets.get_adult(foreground_examples, background_examples),
        "breast_cancer": datasets.get_breast_cancer(foreground_examples, background_examples),
    }
    algorithms_set = {
        "MC-antithetic": algorithms.MonteCarloAntithetic(),
        "Herding": algorithms.KernelHerding(kernel_methods.MallowsKernel()),
        "SBQ": algorithms.SequentialBayesianQuadrature(kernel_methods.MallowsKernel()),
        "Orthogonal": algorithms.OrthogonalSphericalCodes(),
        "Sobol": algorithms.Sobol(),
    }

    df = run_experiments(datasets_set, algorithms_set, repeats, max_evals)
    plot_experiments("new/new", df)


def get_discrepancy(n, d, alg, kernel):
    return kernel_methods.discrepancy(*alg(n, d), kernel)


@mem.cache
def run_discrepancy_experiments(lengths, sizes, repeats):
    algs = {
        "MC-Antithetic": lambda n, d: (algorithms.get_antithetic_permutations(n, d), None),
        "Herding": lambda n, d: (
            kernel_methods.kernel_herding(n, d, kernel_methods.MallowsKernel(), 25), None),
        "SBQ": lambda n, d: kernel_methods.sequential_bayesian_quadrature(n, d, kernel, 25),
        "Orthogonal": lambda n, d: (algorithms._orthogonal_permutations(n, d), None),
        "Sobol": lambda n, d: (sobol_sphere.sobol_permutations(n, d), None),
    }
    df = pd.DataFrame(columns=["Algorithm", "d", "n", "Discrepancy", "std"])
    kernel = kernel_methods.MallowsKernel()

    for d in lengths:
        for n in sizes:
            for name, alg in algs.items():
                if name == "SBQ" and n > 100:
                    df = df.append(
                        {"Algorithm": name, "d": d, "n": n, "Discrepancy": "-", "std": "-"},
                        ignore_index=True)
                    continue

                disc = []
                with futures.ThreadPoolExecutor() as executor:
                    for result in executor.map(get_discrepancy, repeat(n, repeats), repeat(d),
                                               repeat(alg), repeat(kernel)):
                        disc.append(result)
                df = df.append(
                    {"Algorithm": name, "d": d, "n": n, "Discrepancy": np.mean(disc),
                     "std": np.std(disc)},
                    ignore_index=True)
                print(df.to_latex(index=False))
    return df


def discrepancy_experiments():
    lengths = [10, 50, 200]
    sizes = [10, 100, 1000]
    repeats = 25
    df = run_discrepancy_experiments(lengths, sizes, repeats)
    df = df.pivot(index="Algorithm", columns=['d', 'n'], values=['Discrepancy'])
    df = df.sort_index(axis=1)
    df = df.transpose().droplevel(0)

    print(df.to_latex( multirow=True))


kernel_experiments()
kernel_argmax_experiments()
new_experiments()
discrepancy_experiments()
