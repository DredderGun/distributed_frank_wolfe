import random

import numpy as np


def get_matrices_deviation(matrices):
    """
    To reduce similarity value:
    - used 1 norm instead 2 norm
    - use A - \sum A instead A^T A - \sum A^T A (by definition second way is more correct,
    but here we want only matrix deviation)
    """
    matrices_sum = np.sum(matrices, axis=0)
    return np.max(np.array([np.linalg.norm(mx - matrices_sum) for mx in matrices])) / 2


def save_matrices(matrices, filename):
    np.save(filename, matrices)


def load_matrices(filename):
    return np.load(filename, allow_pickle=True)


class ErdosRenyiGraph:
    def __init__(self, num_nodes, prob_edge):
        self.__adj_matrix = None
        self.num_nodes = num_nodes
        self.prob_edge = prob_edge

    def generate_edges(self, normalize_rows=False):
        self.__adj_matrix = [[0] * self.num_nodes for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if i == j or random.random() < self.prob_edge:
                    self.__adj_matrix[i][j] = 1
                    self.__adj_matrix[j][i] = 1

        if normalize_rows:
            self.__normalize_rows()

        self.__adj_matrix = np.array(self.__adj_matrix)

    def __normalize_rows(self):
        for i in range(self.num_nodes):
            row_sum = sum(self.__adj_matrix[i])
            if row_sum > 0:
                self.__adj_matrix[i] = [x / row_sum for x in self.__adj_matrix[i]]

    def get_adj_mx(self):
        assert self.__adj_matrix is not None, "Call __normalize_rows before"
        assert isinstance(self.__adj_matrix, np.ndarray), "Adjacency matrix should be numpy array"
        return self.__adj_matrix

    def display(self):
        for row in self.__adj_matrix:
            print(" ".join(map(str, row)))


def random_point_in_l2_ball(center, radius, pos_dir=False, surface_point=False):
    # Generate a random point on the unit sphere
    ndim = len(center)
    random_direction = np.random.randn(ndim)
    random_direction /= np.linalg.norm(random_direction)

    if surface_point:
        return center + radius * random_direction

    if pos_dir:
        random_direction = np.sign(random_direction) * random_direction

    # Generate a random radius within the given ball's radius
    random_radius = np.random.uniform(radius*0.1, radius*0.2)

    # Scale the random point by the random radius
    random_point = center + random_radius * random_direction

    assert np.linalg.norm(random_point - center) - radius <= 1e-15

    return random_point


def generate_matrix_A(cov_mx, N):
    """
    Generate matrix A with random distribution

    Inputs:
        d - rows
        n - columns

    Returns:
        A:  result matrix
    """
    # Generate samples from a multivariate normal distribution
    mean = np.zeros(cov_mx.shape[0])
    A = np.random.multivariate_normal(mean, cov_mx, size=N)
    return A


def generate_covariance_matrix(d, low, high):
    # Generate eigenvalues uniformly distributed in [low, high]
    eigenvalues = np.random.uniform(low, high, size=d)

    # Generate eigenvectors using QR decomposition of a random matrix
    random_matrix = np.random.randn(d, d)
    q, _ = np.linalg.qr(random_matrix)

    # Construct covariance matrix using eigenvalue decomposition
    covariance_matrix = np.sum([(eigenvalues[j] * np.outer(q[:, j], q[:, j])) for j in range(d)], axis=0)

    return covariance_matrix


if __name__ == "__main__":
    pass
