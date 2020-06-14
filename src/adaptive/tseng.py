import numpy as np
import time
from typing import Callable, TypeVar, Tuple
from ..utility import norm

T = TypeVar('T')


def adaptive_tseng(x_initial: T,
                   tau: float,
                   lambda_initial: float,
                   tolerance: float = 1e-6,
                   max_iterations: int = 1e3,
                   operator: Callable[[T], T] = lambda x: x,
                   projector: Callable[[T], T] = lambda x: x,
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
        x_next = y_current - \
            lambda_current * (operator(y_current) - operator(x_current))

        # step 3
        if norm(operator(x_current) - operator(y_current)) < tolerance:
            lambda_next = lambda_current
        else:
            lambda_next = min(lambda_current, tau *
                np.linalg.norm(x_current - y_current) /
                np.linalg.norm(operator(x_current) - operator(y_current)))

        # next iteration
        iteration_number += 1
        x_current, x_next = x_next, None
        y_current = None
        lambda_current, lambda_next = lambda_next, None


def cached_adaptive_tseng(x_initial: T,
                          tau: float,
                          lambda_initial: float,
                          tolerance: float = 1e-6,
                          max_iterations: int = 1e3,
                          operator: Callable[[T], T] = lambda x: x,
                          projector: Callable[[T], T] = lambda x: x,
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
        x_next = y_current - \
            lambda_current * (operator_y_current - operator_x_current)

        # step 3
        if norm(operator_x_current - operator_y_current) < tolerance:
            lambda_next = lambda_current
        else:
            lambda_next = min(lambda_current, tau *
                np.linalg.norm(x_current - y_current) /
                np.linalg.norm(operator_x_current - operator_y_current))
        
        # next iteration
        iteration_number += 1
        x_current, x_next = x_next, None
        y_current = None
        lambda_current, lambda_next = lambda_next, None
        operator_x_current, operator_y_current = None, None
