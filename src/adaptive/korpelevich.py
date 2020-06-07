import numpy as np
import time
from typing import Callable, TypeVar, Tuple
from ..utility import norm

T = TypeVar('T')


def adaptive_korpelevich(x_initial: T,
                         tau: float,
                         lambda_initial: float,
                         operator: Callable[[T], T],
                         projector: Callable[[T], T],
                         tolerance: float = 1e-6,
                         max_iterations: int = 1e3,
                         **kwargs) -> Tuple[T, int, float]:
    start = time.time()

    # initialization
    iteration_number = 1
    x_current, x_next = x_initial, None
    y_current = None
    lambda_current, lambda_next = lambda_initial, None

    while True:
        # step 1
        y_current = projector(x_current - lambda_current * operator(x_current))

        # stopping criterion
        if norm(x_current - y_current) < tolerance or \
            iteration_number == max_iterations:
            end = time.time()
            duration = end - start
            return x_current, iteration_number, duration

        # step 2
        x_next = projector(x_current - lambda_current * operator(y_current))

        # step 3
        if (operator(x_current) - operator(y_current)).dot(x_next - y_current) <= 0:
            lambda_next = lambda_current
        else:
            lambda_next = min(lambda_current, tau / 2 * 
                (np.linalg.norm(x_current - y_current) ** 2 + 
                 np.linalg.norm(x_next - y_current) ** 2) /
                    (operator(x_current) - operator(y_current)).dot(x_next - y_current))
        
        # next iteration
        iteration_number += 1
        x_current, x_next = x_next, None
        y_current = None
        lambda_current, lambda_next = lambda_next, None


def cached_adaptive_korpelevich(x_initial: T,
                                tau: float,
                                lambda_initial: float,
                                operator: Callable[[T], T],
                                projector: Callable[[T], T],
                                tolerance: float = 1e-6,
                                max_iterations: int = 1e3,
                                **kwargs) -> Tuple[T, int, float]:
    start = time.time()

    # initialization
    iteration_number = 1
    x_current, x_next = x_initial, None
    y_current = None
    lambda_current, lambda_next = lambda_initial, None
    operator_x_current, operator_y_current = None, None

    while True:
        # step 1
        operator_x_current = operator(x_current)
        y_current = projector(x_current - lambda_current * operator_x_current)

        # stopping criterion
        if norm(x_current - y_current) < tolerance or \
            iteration_number == max_iterations:
            end = time.time()
            duration = end - start
            return x_current, iteration_number, duration

        # step 2
        operator_y_current = operator(y_current)
        x_next = projector(x_current - lambda_current * operator_y_current)

        # step 3
        product = (operator_x_current - operator_y_current).dot(x_next - y_current)
        if product <= 0:
            lambda_next = lambda_current
        else:
            lambda_next = min(lambda_current, tau / 2 *
                (np.linalg.norm(x_current - y_current) ** 2 + 
                 np.linalg.norm(x_next - y_current) ** 2) / product)
        
        # next iteration
        iteration_number += 1
        x_current, x_next = x_next, None
        y_current = None
        lambda_current, lambda_next = lambda_next, None
        operator_x_current, operator_y_current = None, None
