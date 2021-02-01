import numpy as np
from numba import njit
import sklearn.neighbors
import math
from qmc import sobol_permutations
from tqdm import tqdm
import heapq


class MaxHeap:
    pq = [[0.0, 0, -1]]
    REMOVED = -1
    entry_map = {}
    counter = 0

    def remove(self, idx):
        entry = self.entry_map.pop(idx)
        entry[-1] = self.REMOVED

    def push(self, weight, idx):
        if idx in self.entry_map:
            self.remove(idx)
        self.counter += 1
        count = self.counter
        entry = [- weight, count, idx]
        self.entry_map[idx] = entry
        heapq.heappush(self.pq, entry)

    def update(self, weight_delta, idx):
        if idx in self.entry_map:
            entry = self.entry_map[idx]
            new_weight = (- entry[0]) + weight_delta
            self.push(new_weight, idx)

    def pop(self):
        while self.pq:
            priority, count, idx = heapq.heappop(self.pq)
            if idx is not self.REMOVED:
                del self.entry_map[idx]
                return -priority, idx
        raise KeyError('pop from an empty priority queue')


def test_max_heap():
    heap = MaxHeap()
    heap.push(2.0, 0)
    heap.push(3.0, 1)
    heap.push(2.5, 2)

    assert heap.pop()[1] == 1
    assert heap.pop()[1] == 2
    assert heap.pop()[1] == 0

    heap.push(2.0, 0)
    heap.push(3.0, 1)
    heap.push(2.5, 2)
    heap.push(4.0, 0)
    assert heap.pop()[1] == 0
    assert heap.pop()[1] == 1
    assert heap.pop()[1] == 2
    try:
        heap.pop()
        assert False
    except:
        pass

    heap.push(2.0, 0)
    heap.push(3.0, 1)
    heap.update(2.0, 0)
    w, idx = heap.pop()
    assert w == 4.0
    assert idx == 0


@njit
def inverse_permutation(x):
    x_inv = np.zeros_like(x)
    for i in range(x.shape[0]):
        x_inv[x[i]] = i
    return x_inv


@njit
def kendall_tau(a, b):
    return np.abs(inverse_permutation(a) - inverse_permutation(b)).sum()


def uniform_random_permutations(n_samples, n_features):
    p = np.zeros((n_samples, n_features), dtype=np.int64)
    for i in range(n_samples):
        p[i] = np.random.permutation(n_features)
    return p


def permutation_singleton_bound(n, d):
    assert d > (n - 1)
    x = 3.0 / 2 + np.sqrt(n * (n - 1) - 2 * d + 1.0 / 4)
    return math.gamma(np.floor(x) + 1)


def get_rmin_rmax(n_features, n_samples, M_size):
    # start with the largest possible kt distance
    d = n_features * (n_features - 1) / 2
    while d > 0:
        if d > (n_features - 1):
            break
        max_samples = permutation_singleton_bound(n_features, d)
        # print("n {} r {} max_samples {}".format(n_features, d / 2, max_samples))
        if max_samples >= n_samples:
            break
        d -= 1
    rmax = d / 2
    return (rmax * (1.0 - np.power(n_samples / M_size, 1.5)) * 0.65, rmax)


@njit
def weight_function(rmin, rmax, dist):
    dist = min(dist, 2 * rmax)
    return np.power(1 - (dist / (2 * rmax)), 8)


def sample_elimination_permutations(n_samples, n_features, M_size=None, bound='sphere'):
    if M_size == None:
        M_size = n_samples * 10
    assert M_size >= n_samples
    M = uniform_random_permutations(M_size, n_features)
    M_inverse = np.zeros_like(M)
    for i in range(M.shape[0]):
        M_inverse[i] = inverse_permutation(M[i])
    tree = sklearn.neighbors.KDTree(M_inverse, metric='manhattan')
    rmin, rmax = get_rmin_rmax(n_features, n_samples, M_size)
    current_n_samples = M_size
    heap = MaxHeap()
    # Initialise weights
    for i in range(M.shape[0]):
        neighbours = tree.query_radius(M_inverse[i:i + 1], r=2 * rmax, return_distance=True)
        w = 0.0
        for j, dist in zip(neighbours[0][0], neighbours[1][0]):
            if i != j:
                w += weight_function(rmin, rmax, dist)
        heap.push(w, i)

    while current_n_samples > n_samples:
        w, j = heap.pop()
        neighbours = tree.query_radius(M_inverse[j:j + 1], r=2 * rmax, return_distance=True)
        for i, dist in zip(neighbours[0][0], neighbours[1][0]):
            if i != j:
                heap.update(-weight_function(rmin, rmax, dist), i)
        current_n_samples -= 1

    selected_samples = [heap.pop()[1] for i in range(n_samples)]
    return M[selected_samples]


def test_inverse():
    p = uniform_random_permutations(100, 50)
    assert np.all(p == map(map(p, inverse_permutation), inverse_permutation))


@njit
def get_distance_matrix(p):
    n = p.shape[0]
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i][j] = kendall_tau(p[i], p[j])
    return distance_matrix


def get_distance_stats(p):
    distance_matrix = get_distance_matrix(p)
    triu = np.triu_indices_from(distance_matrix, k=1)
    max = distance_matrix[triu].max()
    min = distance_matrix[triu].min()
    mean = distance_matrix[triu].mean()
    return min, max, mean


def plot_min_kt():
    import matplotlib.pyplot as plt
    plt.style.use("seaborn")
    plt.rc('font', family='serif')

    np.random.seed(20)
    n_samples = 100
    n_range = [x for x in range(6, 20, 2)]
    sampling_algorithms = {"SampleElimination": sample_elimination_permutations,
                           "UniformRandom": uniform_random_permutations,
                           "Sobol": sobol_permutations}
    for alg_name, alg in sampling_algorithms.items():
        min_kt = []
        for n in tqdm(n_range, desc=alg_name):
            p = alg(n_samples, n)
            min, max, mean = get_distance_stats(p)
            min_kt.append(min)
        plt.plot(n_range, min_kt, label=alg_name)

    plt.xlabel("n")
    plt.ylabel("min kt")
    plt.tight_layout()
    plt.legend()
    plt.show()


import cProfile, pstats, io
from pstats import SortKey

pr = cProfile.Profile()
pr.enable()
plot_min_kt()
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(20)
print(s.getvalue())
