import math
import os

import numpy as np

from functions import FWNodeRelativeSmooth, DistributedRidgeRegressionReferenceFun, RidgeRegression, \
    lmo_l2_ball
from utils import generate_matrix_A, generate_covariance_matrix, load_matrices, save_matrices


def fw_distributed_ridge_regression_problem(d, n, x0, solution, h, graph, radius, node_nmbr=30, noise=0.1, lamda=0,
                                            randseed=-1):
    """
    A ridge regression problem over a network of agents, see description https://arxiv.org/pdf/2110.12347.pdf page 8
    Each node is optimized with Frank-Wolfe method and step with Bregman divergence

    where
        d:  number of rows in data matrix A
        n:  number of cols in data matrix A
        x0: start point
        graph: is an Erdos-Renie graph for network that will optimize function
        comp_nmbr: number of agents in network
        noise:  noise level to generate b = A * x + noise
        lambda: L2 regularization weight

    Return f, h, L, x0:
        f: f(x) = D_KL(Ax, b)
        h: h(x) = Shannon entropy (with L1 regularization as Psi)
        L: L = max(sum(A, axis=0)), maximum column sum
        x0: initial point, scaled version of all-one vector
    """
    assert node_nmbr > 0

    nodes = []
    L = lamda

    filename = "../matrices_regression.npy"
    if os.path.exists(filename):
        matrices = load_matrices(filename)
        print("Matrices loaded from file.")
    else:
        cov_mx = generate_covariance_matrix(d, 1, 1000)
        matrices = []
        for i in range(node_nmbr):
            if randseed > 0:
                np.random.seed(randseed)
            A = generate_matrix_A(cov_mx, n)
            matrices.append(A)
        matrices = np.array(matrices)
        save_matrices(matrices, filename)
        print("Matrices created and saved to file.")

    for i in range(node_nmbr):
        L += matrices[i].T.dot(matrices[i])

    L = np.linalg.norm(L, 'fro') / (node_nmbr * n)

    for i in range(node_nmbr):
        if randseed > 0:
            np.random.seed(randseed)
        A = matrices[i]
        b = np.dot(A, solution) + noise * (np.random.rand(n) - 0.001)
        f = RidgeRegression(A, b, lamda)
        if i == 0 and h is None:
            similarity = math.sqrt((math.log(d / 0.2) / 1))
            h = DistributedRidgeRegressionReferenceFun(f, similarity)
        node = FWNodeRelativeSmooth(f, h, lmo=lmo_l2_ball(radius), L=L)
        nodes.append(node)

    return np.array(nodes)
