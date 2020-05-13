import numpy as np
import time

from typing import Callable, TypeVar
T = TypeVar('T')


def tseng(x_initial: T,
          lambda_: float,
          A: Callable[[T], T],
          ProjectionOntoC: Callable[[T], T],
          tolerance: float = 1e-5,
          max_iterations: int = 1e4,
          debug: bool = False) -> T:
    start = time.time()

    # initialization
    iteration_n = 1
    x_current = x_initial

    while True:
        # step 1
        y_current = ProjectionOntoC(x_current - lambda_ * A(x_current))

        # stopping criterion
        if (np.linalg.norm(x_current - y_current) < tolerance or
            iteration_n == max_iterations):
            if debug:
                end = time.time()
                duration = end - start
                print(f'Took {iteration_n} iterations '
                      f'and {duration:.2f} seconds to converge.')
                return x_current, iteration_n, duration
            return x_current

        # step 2
        x_next = y_current - lambda_ * (A(y_current) - A(x_current))

        # next iteration
        iteration_n += 1
        x_current, x_next = x_next, None
        y_current = None


def cached_tseng(x_initial: T,
                 lambda_: float,
                 A: Callable[[T], T],
                 ProjectionOntoC: Callable[[T], T],
                 tolerance: float = 1e-5,
                 max_iterations: int = 1e4,
                 debug: bool = False) -> T:
    start = time.time()

    # initialization
    iteration_n = 1
    x_current = x_initial

    while True:
        # step 1
        A_x_current = A(x_current)
        y_current = ProjectionOntoC(x_current - lambda_ * A_x_current)

        # stopping criterion
        if (np.linalg.norm(x_current - y_current) < tolerance or
            iteration_n == max_iterations):
            if debug:
                end = time.time()
                duration = end - start
                print(f'Took {iteration_n} iterations '
                      f'and {duration:.2f} seconds to converge.')
                return x_current, iteration_n, duration
            return x_current

        # step 2
        x_next = y_current - lambda_ * (A(y_current) - A_x_current)

        # next iteration
        iteration_n += 1
        x_current, x_next = x_next, None
        y_current = None
