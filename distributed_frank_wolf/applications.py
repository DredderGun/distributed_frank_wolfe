import math
import os

import numpy as np

from functions import FWNodeRelativeSmooth, DistributedRidgeRegressionReferenceFun, RidgeRegression, \
    lmo_l2_ball
from utils import generate_matrix_A, generate_covariance_matrix, load_matrices, save_matrices


def get_matrices_deviation(matrices):
    """
    To reduce similarity value:
    - used 1 norm instead 2 norm
    - use A - \sum A instead A^T A - \sum A^T A (by definition second way is more correct,
    but here we want only matrix deviation)
    """
    matrices_sum = np.sum(matrices, axis=0)
    return np.max(np.array([np.linalg.norm(mx - matrices_sum) for mx in matrices])) / 2


def fw_distributed_ridge_regression_problem(d, n, solution,
                                            h, radius, cond_nmbr=1000, node_nmbr=30,
                                            noise=0.1, lamda=0, similarity_dec_ratio=1):
    """
    A ridge regression problem over a network of agents, see description https://arxiv.org/pdf/2110.12347.pdf page 8
    Each node is optimized with Frank-Wolfe method and step with Bregman divergence

    where
        d:  number of rows in data matrix A
        n:  number of cols in data matrix A
        solution: solution
        cond_nmbr: conditional number i.e. L / mu
        h: divergence
        radius: l2 ball radius constraint
        node_nmbr: number of agents in network
        noise:  noise level to generate b = A * x + noise
        lambda: L2 regularization weight
        similarity_dec_ratio: how much similarity will be decreased inside algorithm

    Return nodes:
        nodes: nodes of the network
    """
    assert node_nmbr > 0
    assert cond_nmbr >= 1

    nodes = []
    L = lamda
    mu0 = 1
    L0 = mu0 * cond_nmbr

    filename = "../FW_matrices_regression.npy"
    if os.path.exists(filename):
        matrices = load_matrices(filename)
        print("Matrices loaded from file.")
    else:
        cov_mx = generate_covariance_matrix(d, mu0, L0)
        matrices = []
        for i in range(node_nmbr):
            A = generate_matrix_A(cov_mx, n)
            matrices.append(A)
        matrices = np.array(matrices)
        save_matrices(matrices, filename)
        print("Matrices created and saved to file.")

    similarity = get_matrices_deviation(matrices)
    print(f'Max deviation is {similarity}')
    similarity /= similarity_dec_ratio

    for i in range(node_nmbr):
        L += matrices[i].T.dot(matrices[i])

    L = np.linalg.norm(L, 'fro') / (node_nmbr * n)

    for i in range(node_nmbr):
        A = matrices[i]
        b = np.dot(A, solution) + noise * (np.random.rand(n) - 0.001)
        f = RidgeRegression(A, b, lamda)
        if h is None:
            # \todo refactor this! Find way to plug that reference function
            h = DistributedRidgeRegressionReferenceFun(f, similarity)
        node = FWNodeRelativeSmooth(f, h, lmo=lmo_l2_ball(radius), L=L)
        nodes.append(node)

    return np.array(nodes)
