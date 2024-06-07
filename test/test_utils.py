from unittest import TestCase

import numpy as np

from distributed_frank_wolf.utils import generate_covariance_matrix, generate_matrix_A, get_matrices_deviation


class Test(TestCase):
    def test_generate_matrix_a(self):
        np.random.seed()
        d = 100
        n = 50
        node_nmbr = 5

        matrices = []
        A = np.random.rand(n, d)
        for i in range(node_nmbr):
            matrices.append(A)
        matrices_same = np.array(matrices)

        mu0 = 1
        L0 = 50
        cov_mx = generate_covariance_matrix(d, mu0, L0)
        matrices = []
        for i in range(node_nmbr):
            A = generate_matrix_A(cov_mx, n)
            matrices.append(A)
        matrices50 = np.array(matrices)

        self.assertEqual(matrices50.shape[0], node_nmbr)

        mu1 = 1
        L1 = 500
        cov_mx = generate_covariance_matrix(d, mu1, L1)
        matrices = []
        for i in range(node_nmbr):
            A = generate_matrix_A(cov_mx, n)
            matrices.append(A)
        matrices500 = np.array(matrices)

        mu2 = 1
        L2 = 1000
        cov_mx = generate_covariance_matrix(d, mu2, L2)
        matrices = []
        for i in range(node_nmbr):
            A = generate_matrix_A(cov_mx, n)
            matrices.append(A)
        matrices1000 = np.array(matrices)

        matrices_same = get_matrices_deviation(matrices_same)
        dev_50 = get_matrices_deviation(matrices50)
        dev_500 = get_matrices_deviation(matrices500)
        dev_1000 = get_matrices_deviation(matrices1000)

        self.assertTrue(dev_1000 > dev_500)
        self.assertTrue(dev_500 > dev_50)
        self.assertTrue(dev_50 > matrices_same)
