import numpy as np
import time
from typing import Callable, TypeVar, Tuple
from ..utility import norm

T = TypeVar('T')


def adaptive_popov(x_initial: T,
                   y_initial: T,
                   tau: float,
                   lambda_initial: float,
                   tolerance: float = 1e-6,
                   max_iterations: int = 1e4,
                   operator: Callable[[T], T] = lambda x: x,
                   projector: Callable[[T], T] = lambda x: x,
                   **kwargs) -> Tuple[T, int, float]:
    start = time.time()

    # initialization
    iteration_number = 1
    x_current, x_next = x_initial, None
    y_previous, y_current = y_initial, None
    lambda_current, lambda_next = lambda_initial, None

    while True:
        # step 1
        y_current = projector(x_current - lambda_current * operator(y_previous))
        
        # step 2
        x_next = projector(x_current - lambda_current * operator(y_current))

        # stopping criterion
        if norm(x_current - y_current) < tolerance and \
            norm(x_next - y_current) < tolerance or \
            iteration_number == max_iterations:
            end = time.time()
            duration = end - start
            return x_current, iteration_number, duration

        # step 3
        if (operator(y_previous) - operator(y_current)).dot(x_next - y_current) <= 0:
            lambda_next = lambda_current
        else:
            lambda_next = min(lambda_current, tau / 2 *
                (norm(y_previous - y_current) ** 2 +
                 norm(x_next - y_current) ** 2) /
                    (operator(y_previous) - operator(y_current)).dot(x_next - y_current))
        
        # next iteration
        iteration_number += 1
        x_current, x_next = x_next, None
        y_previous, y_current = y_current, None
        lambda_current, lambda_next = lambda_next, None


def cached_adaptive_popov(x_initial: T,
                          y_initial: T,
                          tau: float,
                          lambda_initial: float,
                          tolerance: float = 1e-6,
                          max_iterations: int = 1e4,
                          operator: Callable[[T], T] = lambda x: x,
                          projector: Callable[[T], T] = lambda x: x,
                          **kwargs) -> Tuple[T, int, float]:
    start = time.time()

    # initialization
    iteration_number = 1
    x_current, x_next = x_initial, None
    y_previous, y_current = y_initial, None
    lambda_current, lambda_next = lambda_initial, None
    operator_y_previous, operator_y_current = operator(y_previous), None

    while True:
        # step 1
        y_current = projector(x_current - lambda_current * operator_y_previous)
        
        # step 2
        operator_y_current = operator(y_current)
        x_next = projector(x_current - lambda_current * operator_y_current)

        # stopping criterion
        if norm(x_current - y_current) < tolerance and \
            norm(x_next - y_current) < tolerance or \
            iteration_number == max_iterations:
            end = time.time()
            duration = end - start
            return x_current, iteration_number, duration

        # step 3
        product = (operator_y_previous - operator_y_current).dot(x_next - y_current)
        if product <= 0:
            lambda_next = lambda_current
        else:
            lambda_next = min(lambda_current, tau / 2 *
                (norm(y_previous - y_current) ** 2 +
                 norm(x_next - y_current) ** 2) / product)
        
        # next iteration
        iteration_number += 1
        x_current, x_next = x_next, None
        y_previous, y_current = y_current, None
        lambda_current, lambda_next = lambda_next, None
        operator_y_previous, operator_y_current = operator_y_current, None
