#!/usr/bin/env python
import numpy as np
from numpy.linalg import norm
import unittest
from typing import Callable, Tuple

def arrow_hurwitz(x0: np.array, y0: np.array, 
        grad1: Callable[[Tuple[np.array, np.array]], np.array],
        grad2: Callable[[Tuple[np.array, np.array]], np.array],
        rho: Callable[[int], float]=lambda _: 1,
        eps: float=1e-5, max_it: int=1e3) -> Tuple[np.array, np.array, int]:
    """
    :param x0: starting state of vector-variable x
    :param y0: starting state of vector-variable y
    :param rho: step size, default constant 1
    :param grad1: grad_x L(x, y)
    :param grad2: grad_y L(x, y)
    :param eps: desired precision, for stopping criterion
    :param max_it: maximal allowed number of iterations
    :return: (x_k, y_k, k) with k big enough for stopping criterion to hold
    """
    x_now, y_now, k = np.copy(x0), np.copy(y0), 0
    while k < max_it:
        k += 1
        x_next = x_now - rho(k) * grad1(x_now, y_now)
        y_next = y_now + rho(k) * grad2(x_now, y_now)
        if norm(x_next - x_now) < eps and norm(y_next - y_now) < eps:
            break
        x_now, y_now = np.copy(x_next), np.copy(y_next)
    return x_next, y_next, k


class TestArrowHurwitz(unittest.TestCase):
    def test_divergence(self):
        x0, y0 = np.array([1.]), np.array([1.])

        def grad1(x: np.array, y: np.array) -> np.array:
            return y

        def grad2(x: np.array, y: np.array) -> np.array:
            return x

        xn, yn, n = arrow_hurwitz(x0, y0, grad1, grad2, max_it=1e3)

        print(f'x_{n} = {xn}, y_{n} = {yn}')

        self.assertGreater(np.linalg.norm(xn), 1e100)
        self.assertGreater(np.linalg.norm(yn), 1e100)


if __name__ == "__main__":
    unittest.main()
