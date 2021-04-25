import pytest
import numpy as np
import kernel_methods

kernels = [kernel_methods.KTKernel(), kernel_methods.MallowsKernel(l=0.1),
           kernel_methods.MallowsKernel(l=1.0), kernel_methods.SpearmanKernel(),
           kernel_methods.NormalisedMallowsKernel(l=4.5)]


@pytest.mark.parametrize("kernel", kernels)
def test_expected_value(kernel):
    import itertools
    d_tests = [3, 4, 5]
    for d in d_tests:
        p = np.random.permutation(d)
        values = list(map(lambda x: kernel(p, np.array(x)), itertools.permutations(p)))
        expected = np.mean(values)
        assert np.isclose(expected, kernel.expected_value(d))


@pytest.mark.parametrize("kernel", kernels)
def test_positive_definite(kernel):
    import itertools
    d_tests = [3, 4, 5]
    for d in d_tests:
        p = np.array([x for x in itertools.permutations(np.arange(d))])
        n = p.shape[0]
        gram = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                gram[i][j] = kernel(p[i], p[j])
        assert np.all(np.linalg.eigvals(gram) > -1e-8)
