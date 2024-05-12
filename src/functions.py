import numpy as np


class RSmoothFunction:
    """
    Relatively-Smooth Function, can query f(x) and gradient
    """

    def __call__(self, x):
        assert 0, "RSmoothFunction: __call__(x) is not defined"

    def gradient(self, x):
        assert 0, "RSmoothFunction: gradient(x) is not defined"

    def func_grad(self, x, flag):
        """
        flag=0: function, flag=1: gradient, flag=2: function & gradient
        """
        assert 0, "RSmoothFunction: func_grad(x, flag) is not defined"


class Node:
    def get_next_x(self, x, grads, linesearch=False):
        pass

    def get_grad(self, x):
        pass


class LegendreFunction(Node):
    """
    Function of Legendre type, used as the kernel of Bregman divergence for
    composite optimization
         minimize_{x in C} f(x) + Psi(x)
    where f is L-smooth relative to a Legendre function h(x),
          Psi(x) is an additional simple convex function.
    """

    def __call__(self, x):
        assert 0, "LegendreFunction: __call__(x) is not defined."

    def extra_Psi(self, x):
        return 0

    def gradient(self, x):
        assert 0, "LegendreFunction: gradient(x) is not defined."

    def divergence(self, x, y):
        """
        Return D(x,y) = h(x) - h(y) - <h'(y), x-y>
        """
        assert 0, "LegendreFunction: divergence(x,y) is not defined."

    def prox_map(self, g, L):
        """
        Return argmin_{x in C} { Psi(x) + <g, x> + L * h(x) }
        """
        assert 0, "LegendreFunction: prox_map(x, L) is not defined."

    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x in C} { Psi(x) + <g, x> + L * D(x,y)  }
        default implementation by calling prox_map(g - L*g(y), L)
        """
        assert y.shape == g.shape, "Vectors y and g should have same size."
        assert L > 0, "Relative smoothness constant L should be positive."
        return self.prox_map(g - L * self.gradient(y), L)


class FWNodeRelativeSmooth(Node):
    def __init__(self, f, div, lmo, L, gamma=2):
        assert f is not None
        assert div is not None
        assert lmo is not None

        self.__prev_grad = None
        self.div = div
        self.lmo = lmo
        self.f = f
        self.L = L
        self.gamma = gamma

    def get_grad(self, x):
        return self.f.func_grad(x, flag=1)

    def get_next_x(self, x, approx_grad, linesearch=False):
        if linesearch:
            self.L = self.L / 2

        s_k = self.lmo(approx_grad)
        d_k = s_k - x
        div = self.div.divergence(s_k, x)

        grad_d_prod = np.dot(approx_grad, d_k)

        fx, grad = self.f.func_grad(x)
        while True:
            alpha_k = min((-grad_d_prod / (2 * self.L * div)) ** (1 / (self.gamma - 1)), 1)
            x1 = x + alpha_k * d_k
            if not linesearch:
                break
            if self.f.func_grad(x1, flag=0) <= fx + alpha_k * grad_d_prod + alpha_k ** self.gamma * self.L * div:
                break
            self.L = self.L * 2
        x = x1

        return x


class RidgeRegression(RSmoothFunction):
    """
    \\ todo
    """

    def __init__(self, A, b, lamda):
        assert A.shape[0] == b.shape[0], "A and b sizes not matching"
        self.A = A
        self.b = b
        self.lamda = lamda
        self.n = A.shape[0]
        self.d = A.shape[1]

    def __call__(self, x):
        return self.func_grad(x, flag=0)

    def gradient(self, x):
        return self.func_grad(x, flag=1)

    def func_grad(self, x, flag=2):
        assert x.size == self.d, "RidgeRegression: x.size not equal to n."
        Ax = np.dot(self.A, x)
        if flag == 0:
            fx = (1 / (2 * self.n)) * np.linalg.norm(Ax - self.b) ** 2 + self.lamda * np.linalg.norm(x) ** 2
            return fx

        g = (1 / self.n) * np.dot(self.A.T, (Ax - self.b)) + 2 * self.lamda * x
        if flag == 1:
            return g

        # return both function value and gradient
        fx = (1 / (2 * self.n)) * sum((Ax - self.b) ** 2) + self.lamda * np.sum(x ** 2)
        return fx, g


class DistributedRidgeRegressionReferenceFun(LegendreFunction):
    """
    \\ todo
    """

    def __init__(self, f, similarity):
        assert similarity > 0, "similarity constant should be positive"
        self.similarity = similarity
        self.f = f

    def __call__(self, x):
        return self.f(x) + 0.5 * self.similarity * np.linalg.norm(x) ** 2

    def divergence(self, x, y):
        """
        Return D(x,y) = h(x) - h(y) - <h'(y), x-y>
        """
        return self(x) - self(y) - np.dot(self.gradient(y), x - y)

    def gradient(self, x):
        return self.f.gradient(x) + self.similarity * x


class SquaredL2Norm(LegendreFunction):
    """
    h(x) = (1/2)||x||_2^2
    """

    def __call__(self, x):
        return 0.5 * np.dot(x, x)

    def gradient(self, x):
        return x

    def divergence(self, x, y):
        assert x.shape == y.shape, "SquaredL2Norm: x and y not same shape."
        xy = x - y
        return 0.5 * np.dot(xy, xy)

    def prox_map(self, g, L):
        assert L > 0, "SquaredL2Norm: L should be positive."
        return -(1 / L) * g

    def div_prox_map(self, y, g, L):
        assert y.shape == g.shape and L > 0, "Vectors y and g not same shape."
        return y - (1 / L) * g


## FW LMO ##


def lmo_l2_ball(radius, center=None):
    """
    The Frank-Wolfe lmo function for the l2 ball on x > 0 and
    x \in ||radius - center||_2 <= radius
    """

    def f(g):
        if center is None:
            center_p = np.zeros(g.shape[0])
        else:
            center_p = np.array([center] * g.shape[0])
        s = center_p - radius * g / np.linalg.norm(g)
        s[s == 0] = 1e-20

        assert abs(np.linalg.norm(s - center_p) - radius) <= 1e-12

        return s

    return lambda g: f(g)


if __name__ == "__main__":
    pass
