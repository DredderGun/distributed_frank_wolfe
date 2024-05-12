# SONATA algorithm and Frank-Wolfe step

There is implementation of the SONATA (see https://arxiv.org/pdf/2110.12347 for details) where each node uses Frank-Wolfe algorithm with Bregman divergence: $\alpha_k := \min \{ \left( \frac{- \langle \nabla f(x_k), d_k \rangle }{2 L_k V(x_k + d_k, x_k)} \right) ^{\frac{1}{\gamma - 1}}, 1 }$

The project has Ridge Regression problem, data are prepared step by step as it described in https://arxiv.org/pdf/2110.12347 p.8 (**Ridge Regression**)

### How to see something?

You can start with main.py to see some algoritms comparisons
