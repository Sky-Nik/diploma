import numpy as np
import time
from typing import Callable, TypeVar, Tuple
from ..utility import norm

T = TypeVar('T')


def popov(x_initial: T,
          y_initial: T,
          lambda_: float,
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

    while True:
        # step 1
        y_current = projector(x_current - lambda_ * operator(y_previous))
        
        # step 2
        x_next = projector(x_current - lambda_ * operator(y_current))

        # stopping criterion
        if norm(x_current - y_current) < tolerance and \
            norm(x_next - y_current) < tolerance or \
            iteration_number == max_iterations:
            end = time.time()
            duration = end - start
            return x_current, iteration_number, duration

        # next iteration
        iteration_number += 1
        x_current, x_next = x_next, None
        y_previous, y_current = y_current, None


def cached_popov(x_initial: T,
                 y_initial: T,
                 lambda_: float,
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
    operator_y_previous, operator_y_current = operator(y_previous), None

    while True:
        # step 1
        y_current = projector(x_current - lambda_ * operator_y_previous)
        
        # step 2
        operator_y_current = operator(y_current)
        x_next = projector(x_current - lambda_ * operator_y_current)

        # stopping criterion
        if norm(x_current - y_current) < tolerance and \
            norm(x_next - y_current) < tolerance or \
            iteration_number == max_iterations:
            end = time.time()
            duration = end - start
            return x_current, iteration_number, duration

        # next iteration
        iteration_number += 1
        x_current, x_next = x_next, None
        y_previous, y_current = y_current, None
        operator_y_previous, operator_y_current = operator_y_current, None
