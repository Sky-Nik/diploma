import numpy as np
import time
from typing import Callable, TypeVar, Tuple
from ..utility import norm

T = TypeVar('T')


def tseng(x_initial: T,
          lambda_: float,
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

    while True:
        # step 1
        y_current = projector(x_current - lambda_ * operator(x_current))

        # stopping criterion
        if norm(x_current - y_current) < tolerance or \
            iteration_number == max_iterations:
            end = time.time()
            duration = end - start
            return x_current, iteration_number, duration

        # step 2
        x_next = y_current - \
            lambda_ * (operator(y_current) - operator(x_current))

        # next iteration
        iteration_number += 1
        x_current, x_next = x_next, None
        y_current = None


def cached_tseng(x_initial: T,
                 lambda_: float,
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

    while True:
        # step 1
        operator_x_current = operator(x_current)
        y_current = projector(x_current - lambda_ * operator_x_current)

        # stopping criterion
        if norm(x_current - y_current) < tolerance or \
            iteration_number == max_iterations:
            end = time.time()
            duration = end - start
            return x_current, iteration_number, duration

        # step 2
        x_next = y_current - \
            lambda_ * (operator(y_current) - operator_x_current)

        # next iteration
        iteration_number += 1
        x_current, x_next = x_next, None
        y_current = None
