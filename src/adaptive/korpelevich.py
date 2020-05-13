import numpy as np
import time

from typing import Callable, TypeVar
T = TypeVar('T')


def adaptive_korpelevich(x_initial: T,
                         tau: float,
                         lambda_initial: float,
                         A: Callable[[T], T],
                         ProjectionOntoC: Callable[[T], T],
                         tolerance: float = 1e-5,
                         max_iterations: int = 1e4,
                         debug: bool = False) -> T:
    start = time.time()

    # initialization
    iteration_n = 1
    x_current = x_initial
    lambda_current = lambda_initial

    while True:
        # step 1
        y_current = ProjectionOntoC(x_current - lambda_current * A(x_current))

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
        x_next = ProjectionOntoC(x_current - lambda_current * A(y_current))

        # step 3
        if (A(x_current) - A(y_current)).dot(x_next - y_current) <= 0:
            lambda_next = lambda_current
        else:
            lambda_next = min(lambda_current, tau / 2 * 
                (np.linalg.norm(x_current - y_current) ** 2 + 
                 np.linalg.norm(x_next - y_current) ** 2) /
                    (A(x_current) - A(y_current)).dot(x_next - y_current))
        
        # next iteration
        iteration_n += 1
        x_current, x_next = x_next, None
        y_current = None
        lambda_current, lambda_next = lambda_next, None


def cached_adaptive_korpelevich(x_initial: T,
                                tau: float,
                                lambda_initial: float,
                                A: Callable[[T], T],
                                ProjectionOntoC: Callable[[T], T],
                                tolerance: float = 1e-5,
                                max_iterations: int = 1e4,
                                debug: bool = False) -> T:
    start = time.time()

    # initialization
    iteration_n = 1
    x_current = x_initial
    lambda_current = lambda_initial
    A_x_current, A_y_current = None, None

    while True:
        # step 1
        A_x_current = A(x_current)
        y_current = ProjectionOntoC(x_current - lambda_current * A_x_current)

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
        A_y_current = A(y_current)
        x_next = ProjectionOntoC(x_current - lambda_current * A_y_current)

        # step 3
        product = (A_x_current - A_y_current).dot(x_next - y_current)
        if product <= 0:
            lambda_next = lambda_current
        else:
            lambda_next = min(lambda_current, tau / 2 *
                (np.linalg.norm(x_current - y_current) ** 2 + 
                 np.linalg.norm(x_next - y_current) ** 2) / product)
        
        # next iteration
        iteration_n += 1
        x_current, x_next = x_next, None
        y_current = None
        lambda_current, lambda_next = lambda_next, None
        A_x_current, A_y_current = None, None
