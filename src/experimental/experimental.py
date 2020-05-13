import numpy as np
import time

from typing import Callable, TypeVar
T = TypeVar('T')


def experimental(x0: T,
                 x1: T,
                 x2: T,
                 x3: T,
                 lambda_: float,
                 A: Callable[[T], T],
                 ProjectionOntoC: Callable[[T], T],
                 tolerance: float = 1e-5,
                 max_iterations: int = 1e4,
                 debug: bool = False) -> T:
    start = time.time()

    # initialization
    iteration_n = 1
    xp3, xp2, xp1, xc, xn = x0, x1, x2, x3, None
    Axp3, Axp2, Axp1, Axc, Axn = A(xp3), A(xp2), A(xp1), A(xc), None

    while True:
        # step
        # xn = ProjectionOntoC(xc - lambda_ * (4 * Axc - 6 * Axp1 + 4 * Axp2 - Axp3))
        xn = ProjectionOntoC(xc - lambda_ * (2 * Axc - 3 * Axp1 + 3 * Axp2 - Axp3))

        # stopping criterion
        if (np.linalg.norm(xp2 - xp3) < tolerance and
            np.linalg.norm(xp1 - xp2) < tolerance and
            np.linalg.norm(xc - xp1) < tolerance and
            np.linalg.norm(xn - xc) < tolerance or
            iteration_n == max_iterations):
            if debug:
                end = time.time()
                duration = end - start
                print(f'Took {iteration_n} iterations '
                      f'and {duration:.2f} seconds to converge.')
                return xc, iteration_n, duration
            return xc

        # next iteration
        iteration_n += 1
        Axp3, Axp2, Axp1, Axc, Axn = Axp2, Axp1, Axc, A(xn), None
        xp3, xp2, xp1, xc, xn = xp2, xp1, xc, xn, None


def adaptive_experimental(x0: T,
                          x1: T,
                          x2: T,
                          x3: T,
                          tau: float,
                          l0: float,
                          l1: float,
                          l2: float,
                          l3: float,
                          A: Callable[[T], T],
                          ProjectionOntoC: Callable[[T], T],
                          tolerance: float = 1e-5,
                          max_iterations: int = 1e4,
                          debug: bool = False) -> T:
    start = time.time()

    # initialization
    iteration_n = 1
    xp3, xp2, xp1, xc, xn = x0, x1, x2, x3, None
    lp3, lp2, lp1, lc, ln = l0, l1, l2, l3, None
    Axp3, Axp2, Axp1, Axc, Axn = A(xp3), A(xp2), A(xp1), A(xc), None

    while True:
        # step 1
        # xn = ProjectionOntoC(xc - (3 * lc + lp1) * Axc + 3 * (lp1 + lp2) * Axp1 - (3 * lp2 + lp3) * Axp2 + lp3 * Axp3)
        xn = ProjectionOntoC(xc - (lc + lp1) * Axc + (2 * lp1 + lp2) * Axp1 - (2 * lp2 + lp3) * Axp2 + lp3 * Axp3)

        # stopping criterion
        if (np.linalg.norm(xp2 - xp3) < tolerance and
            np.linalg.norm(xp1 - xp2) < tolerance and
            np.linalg.norm(xc - xp1) < tolerance and
            np.linalg.norm(xn - xc) < tolerance or
            iteration_n == max_iterations):
            if debug:
                end = time.time()
                duration = end - start
                print(f'Took {iteration_n} iterations '
                      f'and {duration:.2f} seconds to converge.')
                return xc, iteration_n, duration
            return xc

        # step 2
        Axn = A(xn)
        if np.linalg.norm(Axn - Axc) < tolerance:
            ln = lc
        else:
            ln = min(lc, tau * np.linalg.norm(xn - xc) / np.linalg.norm(Axn - Axc))

        # next iteration
        iteration_n += 1
        Axp3, Axp2, Axp1, Axc, Axn = Axp2, Axp1, Axc, Axn, None
        xp3, xp2, xp1, xc, xn = xp2, xp1, xc, xn, None
        lp3, lp2, lp1, lc, ln = lp2, lp1, lc, ln, None
