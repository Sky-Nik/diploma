import numpy as np
import time
from typing import Callable, TypeVar, Tuple
from ..utility import norm

T = TypeVar('T')


def malitskyi_tam(x0_initial: T,
                  x1_initial: T,
                  lambda_: float,
                  tolerance: float = 1e-6,
                  max_iterations: int = 1e4,
                  operator: Callable[[T], T] = lambda x: x,
                  projector: Callable[[T], T] = lambda x: x,
                  **kwargs) -> Tuple[T, int, float]:
    start = time.time()

    # initialization
    iteration_number = 1
    x_previous, x_current, x_next = x0_initial, x1_initial, None

    while True:
        # step
        x_next = projector(x_current - lambda_ * operator(x_current) - 
            lambda_ * (operator(x_current) - operator(x_previous)))

        # stopping criterion
        if norm(x_current - x_previous) < tolerance and \
            norm(x_next - x_current) < tolerance or \
            iteration_number == max_iterations:
            end = time.time()
            duration = end - start
            return x_current, iteration_number, duration

        # next iteration
        iteration_number += 1
        x_previous, x_current, x_next = x_current, x_next, None


def cached_malitskyi_tam(x0_initial: T,
                         x1_initial: T,
                         lambda_: float,
                         tolerance: float = 1e-6,
                         max_iterations: int = 1e4,
                         operator: Callable[[T], T] = lambda x: x,
                         projector: Callable[[T], T] = lambda x: x,
                         **kwargs) -> Tuple[T, int, float]:
    start = time.time()

    # initialization
    iteration_number = 1
    x_previous, x_current, x_next = x0_initial, x1_initial, None
    operator_x_previous, operator_x_current = \
        operator(x_previous), operator(x_current)

    while True:
        # step
        x_next = projector(x_current - lambda_ * operator_x_current -
            lambda_ * (operator_x_current - operator_x_previous))

        # stopping criterion
        if norm(x_current - x_previous) < tolerance and \
            norm(x_next - x_current) < tolerance or \
            iteration_number == max_iterations:
            end = time.time()
            duration = end - start
            return x_current, iteration_number, duration

        # next iteration
        iteration_number += 1
        operator_x_previous, operator_x_current = \
            operator_x_current, operator(x_next)
        x_previous, x_current, x_next = x_current, x_next, None
