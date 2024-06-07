import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from algorithms import sonata_alg
from applications import fw_distributed_ridge_regression_problem
from functions import SquaredL2Norm
from utils import random_point_in_l2_ball, ErdosRenyiGraph


def start():
    matplotlib.rcParams.update({'font.size': 16, 'legend.fontsize': 14, 'font.family': 'serif'})
    np.random.seed(2024)
    max_itrs = 400
    lamda = 0
    d = 1000
    n = 100
    node_nmbr = 30
    radius = 100
    conditional_nmbr = 1000
    # similarity_ratio = 100
    x0 = random_point_in_l2_ball(np.zeros(d), radius, surface_point=True)
    solution = random_point_in_l2_ball(np.zeros(d), radius)

    graph = ErdosRenyiGraph(node_nmbr, 0.2)
    graph.generate_edges(normalize_rows=True)

    filename = "../matrices_regression.npy"
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    axs = axs.flatten()
    for i, lamda in enumerate([0.01, 0.05, 0.5, 0.8]):
        nodes_sq = fw_distributed_ridge_regression_problem(d, n, solution,
                                                           SquaredL2Norm(),
                                                           radius,
                                                           node_nmbr=node_nmbr,
                                                           cond_nmbr=conditional_nmbr,
                                                           noise=0.1,
                                                           lamda=lamda,
                                                           similarity_dec_ratio=conditional_nmbr*0.8,
                                                           filename=filename)
        nodes = fw_distributed_ridge_regression_problem(d, n, solution,
                                                        None,
                                                        radius,
                                                        node_nmbr=node_nmbr,
                                                        cond_nmbr=conditional_nmbr,
                                                        gamma=1.5,
                                                        noise=0.1,
                                                        lamda=lamda,
                                                        similarity_dec_ratio=conditional_nmbr*0.8,
                                                        filename=filename)

        x00_FW_sq, T00_FW_sq, gaps_sq = sonata_alg(nodes_sq, graph, x0, max_itrs, solution, verbskip=50)
        x00_FW, T00_FW, gaps = sonata_alg(nodes, graph, x0, max_itrs, solution, verbskip=50)

        axs[i].set_title(f'Lambda:{lamda}')
        axs[i].plot(np.arange(len(gaps_sq)), gaps_sq, label=r"FW-euklid-SONATA")
        axs[i].plot(np.arange(len(gaps)), gaps, label=r"FW-RS-SONATA")
        axs[i].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
        axs[i].set_xlabel("k")
        axs[i].set_ylabel("Gap: $\sum \| x_i^k - solution \|^2$")
        axs[i].set_yscale('log')

        if os.path.exists(filename):
            os.remove(filename)
            print(f"File '{filename}' has been removed successfully.")
        else:
            print(f"File '{filename}' does not exist.")

        plt.yscale("log")

    plt.tight_layout(w_pad=4)
    plt.savefig('plot.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    start()
