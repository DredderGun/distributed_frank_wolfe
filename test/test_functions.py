import unittest
import math

import numpy as np

from functions import RidgeRegression, lmo_l2_ball, DistributedRidgeRegressionReferenceFun, SquaredL2Norm, \
    FWNodeRelativeSmooth
from utils import random_point_in_l2_ball, generate_matrix_A, generate_covariance_matrix


class TestFunctions(unittest.TestCase):
    def test_fw_NodeRelativeSmooth_with_custom_prox(self):
        d = 5
        n = 3
        noise = 0.01
        cov_mx = generate_covariance_matrix(d, 1, 1000)
        A = generate_matrix_A(cov_mx, n)
        similarity = math.sqrt((math.log(d / 0.2) / 1))

        radius = 100
        lamda = 0.7
        x0 = random_point_in_l2_ball(np.zeros(d), radius)
        solution = random_point_in_l2_ball(np.zeros(d), radius, surface_point=True)

        b = np.dot(A, solution) + noise * (np.random.rand(n) - 0.001)

        f = RidgeRegression(A, b, lamda)
        node = FWNodeRelativeSmooth(f, DistributedRidgeRegressionReferenceFun(f, similarity),
                                    lmo_l2_ball(radius), similarity)
        x = node.get_next_x(x0, np.ones(d), linesearch=True)

        self.assertIsNotNone(x)

    def test_fw_NodeRelativeSmooth_with_SqL2Norm(self):
        d = 5
        n = 3
        noise = 0.01
        cov_mx = generate_covariance_matrix(d, 1, 1000)
        A = generate_matrix_A(cov_mx, n)
        similarity = math.sqrt((math.log(d / 0.2) / 1))

        radius = 100
        lamda = 0.7
        x0 = random_point_in_l2_ball(np.zeros(d), radius)
        solution = random_point_in_l2_ball(np.zeros(d), radius, surface_point=True)

        b = np.dot(A, solution) + noise * (np.random.rand(n) - 0.001)

        f = RidgeRegression(A, b, lamda)
        node = FWNodeRelativeSmooth(f, SquaredL2Norm(), lmo_l2_ball(radius), similarity)
        x = node.get_next_x(x0, np.ones(d), linesearch=True)

        self.assertIsNotNone(x)


if __name__ == "__main__":
    unittest.main()
