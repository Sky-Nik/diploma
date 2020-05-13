import numpy as np
import time

from typing import Callable, TypeVar
T = TypeVar('T')


def malitskyi_tam(x0_initial: T,
                  x1_initial: T,
                  lambda_: float,
                  A: Callable[[T], T],
                  ProjectionOntoC: Callable[[T], T],
                  tolerance: float = 1e-5,
                  max_iterations: int = 1e4,
                  debug: bool = False) -> T:
    start = time.time()

    # initialization
    iteration_n = 1
    x_previous, x_current, x_next = x0_initial, x1_initial, None

    while True:
        # step
        x_next = ProjectionOntoC(x_current - lambda_ * A(x_current) -
                                 lambda_ * (A(x_current) - A(x_previous)))

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

        # next iteration
        iteration_n += 1
        x_previous, x_current, x_next = x_current, x_next, None


def cached_malitskyi_tam(x0_initial: T,
                         x1_initial: T,
                         lambda_: float,
                         A: Callable[[T], T],
                         ProjectionOntoC: Callable[[T], T],
                         tolerance: float = 1e-5,
                         max_iterations: int = 1e4,
                         debug: bool = False) -> T:
    start = time.time()

    # initialization
    iteration_n = 1
    x_previous, x_current, x_next = x0_initial, x1_initial, None
    A_x_previous, A_x_current, A_x_next = A(x_previous), A(x_current), None

    while True:
        # step
        x_next = ProjectionOntoC(x_current - lambda_ * A_x_current -
                                 lambda_ * (A_x_current - A_x_previous))

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

        # next iteration
        iteration_n += 1
        A_x_previous, A_x_current, A_x_next = A_x_current, A(x_next), None
        x_previous, x_current, x_next = x_current, x_next, None
