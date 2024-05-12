import time

import numpy as np


def sonata_alg(nodes, graph, x0, maxitrs, solution, epsilon=1e-14, verbose=True, verbskip=1):
    """
    It is Frank-Wolfe version of SONATA algorithm (see https://doi.org/10.48550/arXiv.1905.02637)

    Inputs:
        f, h, L:  f is L-smooth relative to h, and Psi is defined within h
        x0:       initial point to start algorithm
        maxitrs:  maximum number of iterations
        gamma:    triangle scaling exponent (TSE) for Bregman distance D_h(x,y)
        lmo:      linear minimization oracle
        epsilon:  stop if D_h(z[k],z[k-1]) < epsilon
        linesearch:  whether or not perform line search (True or False)
        ls_ratio: backtracking line search parameter >= 1
        verbose:  display computational progress (True or False)
        verbskip: number of iterations to skip between displays

    Returns (x, Fx, Ls):
        x:  the last iterate of the algorithm
        F:  array storing F(x[k]) for all k
        Ls: array storing local Lipschitz constants obtained by line search
        T:  array storing time used up to iteration k
    """
    if verbose:
        print("\nFW SONATA algorithm")
        print("     k      X_k       time")

    start_time = time.time()
    X = np.array([np.zeros(x0.shape[0])] * maxitrs)
    gaps = np.zeros(maxitrs)
    T = np.zeros(maxitrs)

    x_vals = np.array([np.copy(x0)] * nodes.shape[0])
    local_grads = np.array([node.get_grad(x) for x, node in zip(x_vals, nodes)])
    grads = np.dot(graph.get_adj_mx(), local_grads)

    for k in range(maxitrs):
        T[k] = time.time() - start_time

        for i in range(nodes.shape[0]):
            x_vals[i] = nodes[i].get_next_x(x_vals[i], grads[i], linesearch=True)

        # S.1 and S.2 steps are communication
        # Consensus S.1 (see https://doi.org/10.48550/arXiv.1905.02637 Alg.1)
        x_vals = np.dot(graph.get_adj_mx(), x_vals)

        local_grads_prev = local_grads
        local_grads = np.array([node.get_grad(x) for x, node in zip(x_vals, nodes)])
        
        # Gradient tracking S.2 (see https://doi.org/10.48550/arXiv.1905.02637 Alg.1)
        grads += (local_grads - local_grads_prev)
        grads = np.dot(graph.get_adj_mx(), local_grads)

        X[k] = x_vals.mean(axis=0)
        gaps[k] = np.sum(np.linalg.norm(x_vals - solution, axis=1)**2) / nodes.shape[0]
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:6.1f}".format(k, np.linalg.norm(X[k]), T[k]))

        if k > 0 and np.linalg.norm(X[k] - X[k - 1]) < epsilon:
            break

    X = X[0:k + 1]
    T = T[0:k + 1]
    gaps = gaps[0:k + 1]
    return X, T, gaps
