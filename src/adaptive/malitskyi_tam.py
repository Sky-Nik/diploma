import numpy as np
import time
from typing import Callable, TypeVar, Tuple
from ..utility import norm

T = TypeVar('T')


def adaptive_malitskyi_tam(x0_initial: T,
                           x1_initial: T,
                           tau: float,
                           lambda0_initial: float,
                           lambda1_initial: float,
                           tolerance: float = 1e-6,
                           max_iterations: int = 1e4,
                           operator: Callable[[T], T] = lambda x: x,
                           projector: Callable[[T], T] = lambda x: x,
                           **kwargs) -> Tuple[T, int, float]:
    start = time.time()

    # initialization
    iteration_n = 1
    x_previous, x_current, x_next = x0_initial, x1_initial, None
    lambda_previous, lambda_current, lambda_next = \
        lambda0_initial, lambda1_initial, None

    while True:
        # step 1
        x_next = projector(x_current - lambda_current * operator(x_current) -
            lambda_previous * (operator(x_current) - operator(x_previous)))

        # stopping criterion
        if norm(x_current - x_previous) < tolerance and \
            norm(x_next - x_current) < tolerance or \
            iteration_n == max_iterations:
            end = time.time()
            duration = end - start
            return x_current, iteration_n, duration

        # step 2
        if norm(operator(x_next) - operator(x_current)) < tolerance:
            lambda_next = lambda_current
        else:
            lambda_next = min(lambda_current, tau *
                norm(x_next - x_current) /
                norm(operator(x_next) - operator(x_current)))

        # next iteration
        iteration_n += 1
        x_previous, x_current, x_next = x_current, x_next, None
        lambda_previous, lambda_current, lambda_next = \
            lambda_current, lambda_next, None


def cached_adaptive_malitskyi_tam(x0_initial: T,
                                  x1_initial: T,
                                  tau: float,
                                  lambda0_initial: float,
                                  lambda1_initial: float,
                                  tolerance: float = 1e-6,
                                  max_iterations: int = 1e4,
                                  operator: Callable[[T], T] = lambda x: x,
                                  projector: Callable[[T], T] = lambda x: x,
                                  **kwargs) -> Tuple[T, int, float]:
    start = time.time()

    # initialization
    iteration_n = 1
    x_previous, x_current, x_next = x0_initial, x1_initial, None
    lambda_previous, lambda_current, lambda_next = \
        lambda0_initial, lambda1_initial, None
    operator_x_previous, operator_x_current, operator_x_next = \
        operator(x_previous), operator(x_current), None

    while True:
        # step 1
        x_next = projector(x_current - lambda_current * operator_x_current -
            lambda_previous * (operator_x_current - operator_x_previous))

        # stopping criterion
        if norm(x_current - x_previous) < tolerance and \
            norm(x_next - x_current) < tolerance or \
            iteration_n == max_iterations:
            end = time.time()
            duration = end - start
            return x_current, iteration_n, duration

        # step 2
        operator_x_next = operator(x_next)
        if norm(operator_x_next - operator_x_current) < tolerance:
            lambda_next = lambda_current
        else:
            lambda_next = min(lambda_current, tau *
                norm(x_next - x_current) /
                norm(operator_x_next - operator_x_current))

        # next iteration
        iteration_n += 1
        x_previous, x_current, x_next = x_current, x_next, None
        lambda_previous, lambda_current, lambda_next = \
            lambda_current, lambda_next, None
        operator_x_previous, operator_x_current, operator_x_next = \
            operator_x_current, operator_x_next, None 
