import numpy as np
import time

from typing import Callable, TypeVar
T = TypeVar('T')


def adaptive_malitskyi_tam(x0_initial: T,
                           x1_initial: T,
                           tau: float,
                           lambda0_initial: float,
                           lambda1_initial: float,
                           A: Callable[[T], T],
                           ProjectionOntoC: Callable[[T], T],
                           tolerance: float = 1e-5,
                           max_iterations: int = 1e4,
                           debug: bool = False) -> T:
    start = time.time()

    # initialization
    iteration_n = 1
    x_previous, x_current, x_next = x0_initial, x1_initial, None
    lambda_previous, lambda_current, lambda_next = lambda0_initial, lambda1_initial, None

    while True:
        # step 1
        x_next = ProjectionOntoC(x_current - lambda_current * A(x_current) -
                                 lambda_previous * (A(x_current) - A(x_previous)))

        # stopping criterion
        if (np.linalg.norm(x_current - x_previous) < tolerance and
            np.linalg.norm(x_next - x_current) < tolerance or
            iteration_n == max_iterations):
            if debug:
                end = time.time()
                duration = end - start
                print(f'Took {iteration_n} iterations '
                      f'and {duration:.2f} seconds to converge.')
                return x_current, iteration_n, duration
            return x_current

        # step 2
        if np.linalg.norm(A(x_next) - A(x_current)) < tolerance:
            lambda_next = lambda_current
        else:
            lambda_next = min(lambda_current, tau *
                np.linalg.norm(x_next - x_current) /
                np.linalg.norm(A(x_next) - A(x_current)))

        # next iteration
        iteration_n += 1
        x_previous, x_current, x_next = x_current, x_next, None
        lambda_previous, lambda_current, lambda_next = lambda_current, lambda_next, None


def cached_adaptive_malitskyi_tam(x0_initial: T,
                                  x1_initial: T,
                                  tau: float,
                                  lambda0_initial: float,
                                  lambda1_initial: float,
                                  A: Callable[[T], T],
                                  ProjectionOntoC: Callable[[T], T],
                                  tolerance: float = 1e-5,
                                  max_iterations: int = 1e4,
                                  debug: bool = False) -> T:
    start = time.time()

    # initialization
    iteration_n = 1
    x_previous, x_current, x_next = x0_initial, x1_initial, None
    lambda_previous, lambda_current, lambda_next = lambda0_initial, lambda1_initial, None
    A_x_previous, A_x_current, A_x_next = A(x_previous), A(x_current), None

    while True:
        # step 1
        x_next = ProjectionOntoC(x_current - lambda_current * A_x_current -
                                 lambda_previous * (A_x_current - A_x_previous))

        # stopping criterion
        if (np.linalg.norm(x_current - x_previous) < tolerance and
            np.linalg.norm(x_next - x_current) < tolerance or
            iteration_n == max_iterations):
            if debug:
                end = time.time()
                duration = end - start
                print(f'Took {iteration_n} iterations '
                      f'and {duration:.2f} seconds to converge.')
                return x_current, iteration_n, duration
            return x_current

        # step 2
        A_x_next = A(x_next)
        if np.linalg.norm(A_x_next - A_x_current) < tolerance:
            lambda_next = lambda_current
        else:
            lambda_next = min(lambda_current, tau *
                np.linalg.norm(x_next - x_current) /
                np.linalg.norm(A_x_next - A_x_current))

        # next iteration
        iteration_n += 1
        x_previous, x_current, x_next = x_current, x_next, None
        lambda_previous, lambda_current, lambda_next = lambda_current, lambda_next, None
        A_x_previous, A_x_current, A_x_next = A_x_current, A_x_next, None 
