import unittest

import numpy as np

from algorithms import sonata_alg
from functions import Node
from utils import ErdosRenyiGraph, random_point_in_l2_ball


class TestNode(Node):
    def get_next_x(self, x, grads, linesearch=False):
        return x + 1

    def get_grad(self, x):
        return x


class TestSonataAlg(unittest.TestCase):
    def test_something(self):
        node_nmbr = 5
        max_itrs = 50
        d = 5
        n = 2
        lamda = 0.2

        nodes = np.array([TestNode()]*node_nmbr)
        graph = ErdosRenyiGraph(node_nmbr, 0.7)
        graph.generate_edges()

        radius = 100
        x0 = random_point_in_l2_ball(np.zeros(d), radius)
        solution = random_point_in_l2_ball(np.zeros(d), radius, surface_point=False)

        graph = ErdosRenyiGraph(node_nmbr, 0.6)
        graph.generate_edges(normalize_rows=True)

        self.assertEqual(graph.get_adj_mx().shape[0], node_nmbr)
        self.assertEqual(graph.get_adj_mx().shape[1], node_nmbr)

        X, T, gaps = sonata_alg(nodes, graph, x0, max_itrs, solution, verbskip=50)
        self.assertEqual(X.shape[0], max_itrs)
        self.assertEqual(T.shape[0], max_itrs)
        self.assertEqual(gaps.shape[0], max_itrs)


if __name__ == '__main__':
    unittest.main()
